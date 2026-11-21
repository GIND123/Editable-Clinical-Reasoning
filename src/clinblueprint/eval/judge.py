"""Local LLM judge for diagnosis equivalence (MedCaseReasoning-style accuracy).

Must be validated against physician labels on a public subset before use
(report agreement in the Supplementary). Never use a hosted judge on
credentialed data; this class enforces that via the compliance guard.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from clinblueprint.compliance.guard import assert_local
from clinblueprint.eval.accuracy import string_judge
from clinblueprint.models import prompts as P


class LocalLLMJudge:
    def __init__(self, model, tokenizer, provenance: str = "public", backend: str = "hf", threshold: float = 0.5):
        assert_local(backend, None, provenance)
        self.model, self.tok, self.threshold = model, tokenizer, threshold
        self._fast = string_judge(0.8)
        self._cache: Dict[Tuple[str, str], bool] = {}

    def p_same(self, a: str, b: str) -> float:
        from clinblueprint.models.backbone import chat_prompt
        from clinblueprint.models.scoring import continuation_logprobs

        prompt = chat_prompt(self.tok, [{"role": "user", "content": P.JUDGE_USER.format(a=a, b=b)}])
        lp = continuation_logprobs(self.model, self.tok, prompt, [" yes", " no"], length_normalize=False)
        z = np.exp(lp - lp.max())
        return float(z[0] / z.sum())

    def __call__(self, a: str, b: str) -> bool:
        key = (a.strip().lower(), b.strip().lower())
        if key not in self._cache:
            self._cache[key] = self._fast(a, b) or self.p_same(a, b) >= self.threshold
        return self._cache[key]
