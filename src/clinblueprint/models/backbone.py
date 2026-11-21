"""Backbone loading (HF transformers + PEFT LoRA, optional 4/8-bit) and generation.

Primary backbones (configs/model/*.yaml): Qwen3-4B (circuit-tracer transcoders
available), MedGemma-1.5-4B / Gemma-3-4B (Gemma Scope 2 SAEs/transcoders),
Qwen3-8B (scale check). Verify the exact HF IDs and licences before download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class BackboneConfig:
    name_or_path: str
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    attn_implementation: Optional[str] = "sdpa"  # use "eager" for attention-knockout experiments
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False
    model_class: str = "causal_lm"  # causal_lm | image_text_to_text (multimodal Gemma-3 / MedGemma)
    max_length: int = 8192
    revision: Optional[str] = None


@dataclass
class LoraSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Sequence[str] = field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))


def from_dict(d: Dict[str, Any]) -> BackboneConfig:
    return BackboneConfig(**{k: v for k, v in d.items() if k in BackboneConfig.__dataclass_fields__})


def load_tokenizer(cfg: BackboneConfig):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.name_or_path, revision=cfg.revision, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(cfg: BackboneConfig, lora: Optional[LoraSpec] = None, adapter_path: Optional[str] = None,
               for_training: bool = False):
    import torch
    import transformers

    dtype = getattr(torch, cfg.dtype)
    kw: Dict[str, Any] = {"torch_dtype": dtype, "revision": cfg.revision, "trust_remote_code": cfg.trust_remote_code}
    if cfg.attn_implementation:
        kw["attn_implementation"] = cfg.attn_implementation
    if cfg.load_in_4bit or cfg.load_in_8bit:
        kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit, load_in_8bit=cfg.load_in_8bit,
            bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    else:
        kw["device_map"] = "auto"
    cls = (transformers.AutoModelForImageTextToText if cfg.model_class == "image_text_to_text"
           else transformers.AutoModelForCausalLM)
    model = cls.from_pretrained(cfg.name_or_path, **kw)

    if for_training:
        if cfg.load_in_4bit or cfg.load_in_8bit:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
        elif cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=for_training)
    elif lora is not None:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout, target_modules=list(lora.target_modules),
            task_type="CAUSAL_LM", bias="none"))
    if not for_training:
        model.eval()
    return model


def chat_prompt(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = True,
                enable_thinking: bool = False) -> str:
    try:  # Qwen3 accepts enable_thinking; other templates ignore / reject it
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                             enable_thinking=enable_thinking)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def generate(model, tokenizer, prompts: List[str], *, max_new_tokens: int = 2048, temperature: float = 0.0,
             top_p: float = 1.0, batch_size: int = 4, prefix_allowed_tokens_fn: Optional[Callable] = None,
             max_input_tokens: Optional[int] = None) -> List[str]:
    import torch

    outs: List[str] = []
    device = next(model.parameters()).device
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=max_input_tokens is not None,
                        max_length=max_input_tokens, add_special_tokens=False).to(device)
        gk: Dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.pad_token_id}
        if temperature > 0:
            gk.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gk.update(do_sample=False)
        if prefix_allowed_tokens_fn is not None:
            gk["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        with torch.no_grad():
            seq = model.generate(**enc, **gk)
        for j in range(len(batch)):
            outs.append(tokenizer.decode(seq[j, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def json_prefix_fn(tokenizer, schema: dict):
    """Constrained JSON decoding via lm-format-enforcer (optional dependency)."""
    try:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    except ImportError:
        return None
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema))
