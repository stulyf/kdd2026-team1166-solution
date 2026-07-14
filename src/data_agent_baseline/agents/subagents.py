from __future__ import annotations

import json
import re
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage
from data_agent_baseline.agents.prompt import (
    ANSWER_VERIFIER_PROMPT,
    FORCE_ANSWER_PROMPT,
    PLANNER_PROMPT,
)


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


async def run_planner(
    *,
    model: ModelAdapter,
    question: str,
    context_summary: str,
) -> str:
    """Create a short initial plan for the main ReAct loop."""
    content = (
        f"Question:\n{question}\n\n"
        f"Lightweight context summary:\n{context_summary}\n\n"
        "Create the initial plan now."
    )
    response = await model.complete(
        [
            ModelMessage(role="system", content=PLANNER_PROMPT),
            ModelMessage(role="user", content=content),
        ]
    )
    return response.strip()


async def verify_answer(
    *,
    model: ModelAdapter,
    question: str,
    scratchpad: str,
    columns: list[str],
    rows: list[list[Any]],
    knowledge: str = "",
    provenance: str = "",
    loose_sources: list[str] | None = None,
    reachable: str = "",
) -> dict[str, str]:
    """Check a proposed final answer for SOURCE/COLUMN alignment.

    `knowledge` is the term->authoritative table/column guide, `provenance` lists
    the files/tables the recent steps actually read (so a decoy source is visible),
    `loose_sources` lists top-level pre-computed/decoy files to distrust, and
    `reachable` is the set of tables/files/columns the agent has actually observed
    in this task (so the verifier never demands a knowledge-guide table/column that
    does not physically exist)."""
    preview_rows = rows[:10]
    knowledge_excerpt = (knowledge or "").strip()
    if len(knowledge_excerpt) > 7000:
        knowledge_excerpt = knowledge_excerpt[:7000] + "\n...(truncated)"
    loose = loose_sources or []
    content = (
        f"Question:\n{question}\n\n"
        f"Knowledge guide (authoritative term -> table/column):\n"
        f"{knowledge_excerpt or '(none provided)'}\n\n"
        f"REACHABLE data actually present in this task (tables/files/columns the agent "
        f"has observed - the authoritative source MUST be in here to be demandable):\n"
        f"{reachable or '(none observed yet)'}\n\n"
        f"Loose top-level files that are likely PRE-COMPUTED DECOYS (distrust these):\n"
        f"{', '.join(loose) if loose else '(none)'}\n\n"
        f"Answer provenance (sources the recent steps actually read):\n"
        f"{provenance or '(unknown)'}\n\n"
        f"Working memory:\n{scratchpad or '(empty)'}\n\n"
        "Proposed answer:\n"
        f"{json.dumps({'columns': columns, 'rows_preview': preview_rows, 'row_count': len(rows)}, ensure_ascii=False, indent=2)}"
    )
    response = await model.complete(
        [
            ModelMessage(role="system", content=ANSWER_VERIFIER_PROMPT),
            ModelMessage(role="user", content=content),
        ]
    )
    try:
        parsed = json.loads(_strip_json_fence(response))
    except json.JSONDecodeError:
        return {
            "decision": "accept",
            "reason": "Verifier returned unparsable output; avoid blocking a valid answer.",
            "feedback": "",
        }
    if not isinstance(parsed, dict):
        return {
            "decision": "accept",
            "reason": "Verifier returned non-object output; avoid blocking a valid answer.",
            "feedback": "",
        }
    decision = str(parsed.get("decision", "accept")).strip().lower()
    if decision not in {"accept", "reject"}:
        decision = "accept"
    return {
        "decision": decision,
        "reason": str(parsed.get("reason", "")),
        "feedback": str(parsed.get("feedback", "")),
    }


async def force_final_answer(
    *,
    model: ModelAdapter,
    question: str,
    scratchpad: str,
    knowledge: str,
    result_tables: list[dict[str, Any]],
    python_outputs: list[str],
    video_evidence: str = "",
) -> dict[str, Any] | None:
    """Last-resort answer synthesizer for when the main loop ran out of steps.

    `result_tables` is a list of {index, columns, preview_rows, row_count} blocks
    describing tables the agent already computed; the returned spec may point back
    at one of them via `use_result_index` so the exact rows are reused verbatim.
    Returns the parsed JSON spec, or None if nothing usable was produced."""
    knowledge_excerpt = (knowledge or "").strip()
    if len(knowledge_excerpt) > 5000:
        knowledge_excerpt = knowledge_excerpt[:5000] + "\n...(truncated)"

    tables_text_parts: list[str] = []
    for tbl in result_tables:
        tables_text_parts.append(
            f"[RESULT TABLE index={tbl['index']}] from step {tbl.get('step_index')}, "
            f"total_rows={tbl.get('row_count')}\n"
            f"columns={json.dumps(tbl['columns'], ensure_ascii=False)}\n"
            f"row_preview={json.dumps(tbl.get('preview_rows', []), ensure_ascii=False)}"
        )
    tables_text = "\n\n".join(tables_text_parts) if tables_text_parts else "(no result tables captured)"
    python_text = "\n\n".join(python_outputs) if python_outputs else "(no python output captured)"
    video_text = (video_evidence or "").strip() or "(no video evidence captured)"

    content = (
        f"Question:\n{question}\n\n"
        f"Knowledge guide:\n{knowledge_excerpt or '(none)'}\n\n"
        f"Working memory (scratchpad):\n{scratchpad or '(empty)'}\n\n"
        f"Available RESULT TABLES (prefer reusing one of these):\n{tables_text}\n\n"
        f"Recent Python stdout:\n{python_text}\n\n"
        "Verified VIDEO EVIDENCE (authoritative readings from inspect_video; if no "
        "result table answers the question, the requested values are likely here - "
        "extract them directly, minding any average/category-peak caveats):\n"
        f"{video_text}\n\n"
        "Produce the final answer JSON now. Never emit placeholders like "
        "'需要查询数据'/'unknown' - use the concrete values from the evidence above. "
        "If the answer is a per-group value, include one row for EVERY group shown in "
        "the evidence (do not drop any category, e.g. 博士后/postdoc); a missing row "
        "makes the whole column mismatch and scores zero."
    )
    response = await model.complete(
        [
            ModelMessage(role="system", content=FORCE_ANSWER_PROMPT),
            ModelMessage(role="user", content=content),
        ]
    )
    try:
        parsed = json.loads(_strip_json_fence(response))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed
