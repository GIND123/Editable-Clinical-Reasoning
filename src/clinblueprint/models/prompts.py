"""All prompt templates in one place (versioned; the paper's Supplementary lists them)."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from clinblueprint.schema.serialize import DRAFTER_JSON_SCHEMA

PROMPT_VERSION = "v1"

DRAFTER_SYSTEM = """You are a clinical reasoning assistant. Read the clinical note and write your diagnostic reasoning as a JSON reasoning graph.

Rules:
1. "observations": findings copied VERBATIM from the note (exact substrings, no paraphrase). Give each a polarity (present / absent / uncertain) and the note section.
2. "rationales": each links one or more premises (observation ids, or other rationale ids) to ONE diagnosis id, with relation "supports" or "refutes", a strength in [0,1], and a one-sentence medical claim.
3. "diagnoses": nodes of the diagnostic knowledge graph given below (use the exact category and node names). Use "refines" to point a more specific diagnosis to the more general one it specialises.
4. "must_not_miss": dangerous alternatives you considered, with status excluded / open / not_addressed and the evidence ids.
5. Write the evidence first and the conclusions after it: observations, then rationales, then diagnoses.
6. Output only the JSON object. No text outside it."""

CATEGORY_SELECT = """Read the clinical note and choose up to {k} disease categories from the list that best explain the presentation, most likely first.

Categories:
{categories}

Note:
{note}

Answer with a JSON list of category names only."""

DRAFTER_USER = """Diagnostic knowledge graph (candidate categories; node: diagnostic criteria):
{kg}

Clinical note:
{note}

Return the reasoning graph JSON."""

LINK_SYSTEM = """You update a clinical reasoning graph when a new finding arrives. Decide which candidate diagnoses the finding bears on, in which direction, and how strongly. If it does not bear on any candidate, return []. Output only JSON."""

LINK_USER = """Current reasoning graph:
{graph}

Candidate diagnoses:
{candidates}

New finding ({source}): "{finding}"

Return a JSON list of links: [{{"category": "...", "leaf": "...", "relation": "supports|refutes", "strength": 0.0-1.0, "claim": "..."}}]"""

REPAIR_USER = """The reasoning-graph node below failed a check ({problem}).
Node: {node}

Clinical note:
{note}

Return a corrected JSON object for this single node (same id). If an observation cannot be quoted verbatim from the note, return {{"delete": true}}."""

VERBALIZER_SYSTEM = """You explain a diagnostic reasoning graph to a clinician. Use ONLY the graph below; do not add findings, numbers or diagnoses that are not in it. Cite node ids in brackets, e.g. [O3], [R2]. Be concise: a summary line, the key supporting evidence, the evidence against, and the must-not-miss items with their status."""

VERBALIZER_USER = """Reasoning graph:
{graph}

Readout (probabilities computed from the graph):
{ddx}

Write the explanation."""

COT_SYSTEM = "You are an experienced physician. Reason step by step about the case, then state the single most likely diagnosis."
COT_USER = """Clinical note:
{note}

{candidates_block}Think step by step. End with a line of the form:
Final diagnosis: <diagnosis>"""

CRITIQUE_USER = """Here is a reasoning graph for the note. List concrete problems: unsupported diagnoses, observations not quoted verbatim, missing key findings, wrong relations, unaddressed must-not-miss diagnoses.

Note:
{note}

Graph:
{graph}"""

REVISE_USER = """Revise the reasoning graph to fix these problems. Return only the corrected JSON.

Problems:
{critique}

Note:
{note}

Graph:
{graph}"""

TEACHER_SYSTEM = """You convert a published clinical case and the authors' diagnostic reasoning into a reasoning graph (JSON) that follows the schema exactly. Observations must be exact substrings of the CASE TEXT. Rationales must reflect the authors' reasoning. The final diagnosis must be the most specific diagnosis node. Output only JSON."""

TEACHER_USER = """{kg_block}CASE TEXT:
{case}

AUTHORS' DIAGNOSTIC REASONING:
{reasoning}

FINAL DIAGNOSIS: {final_dx}

Return the reasoning graph JSON."""

NLI_USER = """Premise (from the patient's record):
{premise}

Claim:
{claim}

Does the record entail, contradict, or say nothing about the claim? Answer with one word: entailment, contradiction, or neutral."""

JUDGE_USER = """Are these two diagnoses the same clinical entity (synonyms or one being an acceptable, equally specific name for the other)?
A: {a}
B: {b}
Answer yes or no."""

COUNTERFACTUAL_AGENT_USER = """Clinical note:
{note}

Candidate diagnoses: {candidates}

For each candidate, give a probability (0-1) that it is the correct diagnosis. Return JSON {{"<diagnosis>": p, ...}}."""


def schema_block() -> str:
    return json.dumps(DRAFTER_JSON_SCHEMA)


def drafter_messages(note: str, kg_text: str, include_schema: bool = True) -> List[Dict[str, str]]:
    sys = DRAFTER_SYSTEM + (f"\n\nJSON schema:\n{schema_block()}" if include_schema else "")
    return [{"role": "system", "content": sys}, {"role": "user", "content": DRAFTER_USER.format(kg=kg_text, note=note)}]


def category_messages(note: str, categories: Sequence[str], k: int = 3) -> List[Dict[str, str]]:
    return [{"role": "user", "content": CATEGORY_SELECT.format(k=k, categories="\n".join(categories), note=note)}]


def link_messages(graph_text: str, candidates: Sequence[str], finding: str, source: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": LINK_SYSTEM},
            {"role": "user", "content": LINK_USER.format(graph=graph_text, candidates="\n".join(candidates),
                                                         finding=finding, source=source)}]


def verbalizer_messages(graph_text: str, ddx_text: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": VERBALIZER_SYSTEM},
            {"role": "user", "content": VERBALIZER_USER.format(graph=graph_text, ddx=ddx_text)}]


def cot_messages(note: str, candidates: Optional[Sequence[str]] = None) -> List[Dict[str, str]]:
    block = ("Candidate diagnoses:\n" + "\n".join(f"- {c}" for c in candidates) + "\n\n") if candidates else ""
    return [{"role": "system", "content": COT_SYSTEM},
            {"role": "user", "content": COT_USER.format(note=note, candidates_block=block)}]


def teacher_messages(case: str, reasoning: str, final_dx: str, kg_text: Optional[str]) -> List[Dict[str, str]]:
    kg_block = f"Diagnostic knowledge graph (use these node names when applicable):\n{kg_text}\n\n" if kg_text else ""
    return [{"role": "system", "content": TEACHER_SYSTEM + f"\n\nJSON schema:\n{schema_block()}"},
            {"role": "user", "content": TEACHER_USER.format(kg_block=kg_block, case=case, reasoning=reasoning,
                                                            final_dx=final_dx)}]
