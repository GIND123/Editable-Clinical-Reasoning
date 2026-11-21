"""Data-governance guard rails (docs/COMPLIANCE.md).

PhysioNet credentialed data (MIMIC-IV, MIMIC-IV-Note, DiReCT, MIMIC-CDM,
ER-Reason, MedNLI) and DUA data (SCT-Bench held-out) must never be sent to
a hosted API, judge or simulator. Every code path that sends text to a model
calls ``assert_local(backend, endpoint, provenance)`` first.
"""

from __future__ import annotations

import ipaddress
from typing import Iterable, Optional
from urllib.parse import urlparse

STRICTNESS = {"public": 0, "dua": 1, "physionet": 2}
LOCAL_BACKENDS = {"hf", "transformers", "vllm_local", "static", "mock", "llama_cpp"}
HOSTED_BACKENDS = {"openai", "anthropic", "openrouter", "azure", "gemini", "vertex", "together", "bedrock",
                   "groq", "mistral", "cohere", "fireworks", "deepseek_api"}
HOSTED_HOSTS = ("openai.com", "anthropic.com", "openrouter.ai", "azure.com", "googleapis.com", "together.xyz",
                "amazonaws.com", "groq.com", "mistral.ai", "cohere.ai", "fireworks.ai", "deepseek.com")


class ComplianceError(RuntimeError):
    pass


def is_local_endpoint(endpoint: Optional[str]) -> bool:
    if not endpoint:
        return True  # in-process model
    if endpoint.startswith("unix:") or endpoint.startswith("/"):
        return True
    host = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}").hostname or ""
    if host in ("localhost", "0.0.0.0") or host.endswith(".local"):
        return True
    if any(host == h or host.endswith("." + h) for h in HOSTED_HOSTS):
        return False
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private
    except ValueError:
        return False  # unknown public hostname -> treat as remote


def assert_local(backend: str, endpoint: Optional[str], provenance: str) -> None:
    """Raise unless sending ``provenance`` data to (backend, endpoint) is allowed."""
    level = STRICTNESS.get(provenance, 2)  # unknown provenance -> strictest
    if level == 0:
        return
    b = (backend or "").lower()
    if b in HOSTED_BACKENDS or not is_local_endpoint(endpoint) or (b and b not in LOCAL_BACKENDS):
        raise ComplianceError(
            f"refusing to send {provenance!r} data to backend={backend!r} endpoint={endpoint!r}; "
            "credentialed/DUA text may only be processed by local models (docs/COMPLIANCE.md)"
        )


def strictest_provenance(provenances: Iterable[str]) -> str:
    ps = list(provenances) or ["public"]
    return max(ps, key=lambda p: STRICTNESS.get(p, 2))


def suppress_small_cells(value, min_count: int = 10):
    """PhysioNet/MIMIC rule: never report counts 1..(min_count-1). Works on ints, dicts, DataFrames."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return f"<{min_count}" if 0 < value < min_count else value
    if isinstance(value, dict):
        return {k: suppress_small_cells(v, min_count) for k, v in value.items()}
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return value.apply(lambda col: col.map(
                lambda x: f"<{min_count}" if isinstance(x, (int,)) and not isinstance(x, bool) and 0 < x < min_count
                else x))
    except ImportError:  # pragma: no cover
        pass
    return value
