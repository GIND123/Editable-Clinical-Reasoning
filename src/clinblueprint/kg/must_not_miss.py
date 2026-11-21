"""Clinician-authored must-not-miss lists, one per disease category.

File format (configs/schema/must_not_miss.yaml)::

    status: DRAFT | APPROVED
    approved_by: <initials or empty>
    categories:
      Acute Coronary Syndrome: [Aortic dissection, Pulmonary embolism, ...]

The shipped file is a DRAFT for the clinician co-author to edit and sign off
(see clinician/tasks/01_must_not_miss_lists.md). Experiments refuse to run in
``strict`` mode until ``status: APPROVED``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import yaml


def load_must_not_miss(path: str | Path, strict: bool = False) -> Dict[str, List[str]]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    status = str(data.get("status", "DRAFT")).upper()
    if status != "APPROVED":
        msg = f"must-not-miss list at {path} has status {status}; clinician sign-off required"
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)
    cats = data.get("categories", {}) or {}
    return {str(k): [str(x) for x in (v or [])] for k, v in cats.items()}
