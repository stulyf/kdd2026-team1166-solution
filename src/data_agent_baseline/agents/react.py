from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.agents.subagents import force_final_answer, run_planner, verify_answer
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 70
    enable_planner: bool = True
    enable_answer_verify: bool = True
    recent_full_steps: int = 3


VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi"}


def _find_first_video(context_dir: Path) -> Path | None:
    if not context_dir.exists():
        return None
    for path in sorted(context_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            return path
    return None


def _probe_video_duration(video_path: Path) -> float | None:
    try:
        import av  # noqa: PLC0415

        with av.open(str(video_path)) as container:
            if container.duration is not None:
                return round(float(container.duration) / 1_000_000.0, 1)
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                return round(float(stream.duration * stream.time_base), 1)
    except Exception:  # noqa: BLE001
        return None
    return None


def _build_initial_user_content(task: PublicTask) -> str:
    """Build the initial user message.

    The raw video is intentionally NOT attached here. Instead the agent uses the
    `inspect_video` tool, which runs a separate vision sub-call and returns text.
    This keeps the main conversation free of image/video tokens (avoids context
    explosion and per-step re-sending of the full video).
    """
    task_prompt = build_task_prompt(task)
    video_path = _find_first_video(task.context_dir)
    if video_path is None:
        return task_prompt

    try:
        rel = video_path.relative_to(task.context_dir)
    except ValueError:
        rel = video_path.name
    duration = _probe_video_duration(video_path)
    dur_str = f" (duration ~{duration:.1f}s)" if duration else ""
    return (
        f"{task_prompt}\n\n"
        f"This task includes a briefing video at `{rel}`{dur_str}. The video shows "
        f"rules/thresholds/configurations as on-screen text. Use the `inspect_video` "
        f"tool to watch it and read those values precisely before filtering data."
    )


_RECENT_OBS_CHAR_CAP = 40000
_SCRATCHPAD_CHAR_CAP = 30000


def _truncate_text(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    head = text[: cap - 200]
    return (
        f"{head}\n\n... [observation truncated: {len(text)} chars total, showing first "
        f"{cap - 200}. Re-run the tool with a tighter query/limit if you need the rest.]"
    )


def _shorten(value: Any, cap: int = 120) -> str:
    text = str(value)
    if len(text) <= cap:
        return text
    return f"{text[: cap - 3]}..."


_VIDEO_EVIDENCE_OBS_CAP = 1200
_MEMORY_GREP_TEXT_CAP = 4000


def _step_searchable_text(step: StepRecord) -> str:
    try:
        obs_text = json.dumps(step.observation, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        obs_text = str(step.observation)
    return f"{step.thought}\n{obs_text}"


def _memory_grep(
    steps: list[StepRecord],
    *,
    pattern: str | None,
    step_index: int | None,
    max_matches: int,
) -> dict[str, Any]:
    """Search the agent's own past steps and return the original TEXT (no images).

    This lets the model expand history that the sliding window compressed into a
    one-line summary, without re-introducing raw frames into the main context.
    """
    results: list[dict[str, Any]] = []

    if step_index is not None:
        for step in steps:
            if step.step_index == step_index:
                results.append(
                    {
                        "step": step.step_index,
                        "action": step.action,
                        "thought": _truncate_text(step.thought, _MEMORY_GREP_TEXT_CAP),
                        "observation": _truncate_text(
                            _step_searchable_text(step), _MEMORY_GREP_TEXT_CAP
                        ),
                    }
                )
                break
        return {"status": "ok", "mode": "step_index", "matches": results}

    if not pattern:
        return {"status": "error", "error": "Provide either 'pattern' or 'step_index'."}

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return {"status": "error", "error": f"Invalid regex pattern: {exc}"}

    for step in steps:
        text = _step_searchable_text(step)
        if regex.search(text):
            results.append(
                {
                    "step": step.step_index,
                    "action": step.action,
                    "thought": _truncate_text(step.thought, _MEMORY_GREP_TEXT_CAP),
                    "observation": _truncate_text(text, _MEMORY_GREP_TEXT_CAP),
                }
            )
            if len(results) >= max_matches:
                break

    return {"status": "ok", "mode": "pattern", "pattern": pattern, "match_count": len(results), "matches": results}


def _render_video_evidence(entries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for e in entries:
        conf = e.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
        header = (
            f"- [step {e.get('step')}] window={e.get('window')} confidence={conf_str}"
        )
        query = str(e.get("query") or "").strip()
        if query:
            header += f" query={_shorten(query, 120)}"
        lines.append(header)
        obs = str(e.get("observation") or "").strip()
        if obs:
            lines.append(f"  {_truncate_text(obs, _VIDEO_EVIDENCE_OBS_CAP)}")
    return "\n".join(lines)


def _summarize_step(step: StepRecord) -> str:
    """Build a compact semantic summary for old steps.

    Older raw observations are deliberately not retained in full; durable facts
    should live in the per-turn `scratchpad` field.
    """
    obs = step.observation if isinstance(step.observation, dict) else {}
    content = obs.get("content") if isinstance(obs.get("content"), dict) else {}
    ok = obs.get("ok", step.ok)

    arg_bits: list[str] = []
    if "path" in step.action_input:
        arg_bits.append(f"path={_shorten(step.action_input.get('path'), 80)}")
    if "sql" in step.action_input:
        arg_bits.append(f"sql={_shorten(step.action_input.get('sql'), 140)}")
    if "query" in step.action_input:
        arg_bits.append(f"query={_shorten(step.action_input.get('query'), 120)}")
    if "pattern" in step.action_input:
        arg_bits.append(f"pattern={_shorten(step.action_input.get('pattern'), 80)}")
    result_bits: list[str] = [f"ok={ok}"]
    for key in ("row_count", "match_count", "column_count", "frame_count", "frames_inspected"):
        if key in content:
            result_bits.append(f"{key}={content[key]}")
    if content.get("truncated") is True:
        result_bits.append("truncated=True")
    if "error" in obs:
        result_bits.append(f"error={_shorten(obs.get('error'), 160)}")
    elif "error" in content:
        result_bits.append(f"error={_shorten(content.get('error'), 160)}")

    args = ", ".join(arg_bits)
    result = ", ".join(result_bits)
    args_text = f"({args})" if args else ""
    return f"[step {step.step_index}] {step.action}{args_text} -> {result}"


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _escape_raw_control_chars_in_strings(text: str) -> str:
    """Escape literal newlines/tabs/carriage-returns that appear INSIDE JSON
    string values. LLMs frequently emit multi-line Python in the ``code`` field
    with real newlines instead of ``\\n``, which makes ``json.loads`` fail with
    'Expecting , delimiter'. We only touch control chars within strings, leaving
    structural whitespace untouched."""
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_string = True
    return "".join(out)


def _strip_trailing_garbage(text: str) -> str:
    """Return the substring up to the first balanced top-level JSON object,
    discarding anything after it (stray quotes/braces/commentary LLMs append)."""
    brace_count = 0
    in_string = False
    escaped = False
    for index, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[: index + 1]
    return text


def _fix_bracket_mismatch(text: str) -> str:
    """Repair the most common LLM bracket errors: a missing closing ``]`` in
    ``rows`` (e.g. ``...]}}`` instead of ``...]]}}``) and unbalanced braces."""
    open_brackets = text.count("[")
    close_brackets = text.count("]")
    missing = open_brackets - close_brackets
    if missing > 0:
        for insert_pos in range(len(text) - 1, -1, -1):
            if text[insert_pos] in "}]":
                candidate = text[:insert_pos] + "]" * missing + text[insert_pos:]
                try:
                    json.loads(candidate)
                    text = candidate
                    break
                except json.JSONDecodeError:
                    continue
        else:
            text += "]" * missing

    open_braces = text.count("{")
    close_braces = text.count("}")
    if close_braces < open_braces:
        text += "}" * (open_braces - close_braces)
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    """Decode one JSON object from possibly-malformed model output, applying an
    escalating ladder of repairs (control-char escaping, bracket repair, trailing
    garbage removal) before giving up."""
    candidates = [text]
    escaped = _escape_raw_control_chars_in_strings(text)
    if escaped != text:
        candidates.append(escaped)
    candidates.append(_strip_trailing_garbage(_fix_bracket_mismatch(escaped)))

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            payload, end = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        remainder = candidate[end:].strip()
        if remainder:
            # Tolerate trailing noise (stray braces/brackets/backticks/whitespace);
            # reject only if a genuine second object remains.
            cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder)
            cleaned_remainder = re.sub(r"[}\]`\s]+", "", cleaned_remainder).strip()
            if cleaned_remainder:
                last_error = ValueError("Model response must contain only one JSON object.")
                continue
        if not isinstance(payload, dict):
            raise ValueError("Model response must be a JSON object.")
        return payload

    raise last_error or ValueError("Could not parse a JSON object from the model response.")


def _iter_json_objects(text: str) -> list[dict[str, object]]:
    """Decode every top-level JSON object in ``text``, skipping junk between them.

    Models sometimes split one logical step across SEVERAL JSON objects, e.g.
    ``{"thought":..,"scratchpad":..}`` followed by ``{"action":..,"action_input":..}``.
    The strict single-object loader only sees the first (action-less) object; this
    collects them all so they can be merged."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, object]] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        i = end
    return objects


def _is_empty_value(value: object) -> bool:
    return value is None or value == "" or value == {} or value == []


def _load_merged_json_object(text: str) -> dict[str, object]:
    """Parse the model step, tolerating a step split across multiple JSON objects.

    Tries the strict single-object loader first (unchanged behavior). Only when the
    parsed object carries no usable action/answer does it fall back to decoding all
    top-level objects and merging them (first non-empty value per key wins, so the
    thought/scratchpad from object 1 and the action/action_input from object 2 are
    combined into one step)."""
    single: dict[str, object] | None = None
    try:
        single = _load_single_json_object(text)
    except Exception:  # noqa: BLE001
        single = None

    def _has_command(obj: dict[str, object] | None) -> bool:
        if not obj:
            return False
        return bool(obj.get("action")) or any(
            k in obj for k in ("answer", "value", "result", "count", "data")
        )

    if _has_command(single):
        return single  # type: ignore[return-value]

    escaped = _escape_raw_control_chars_in_strings(text)
    objects = _iter_json_objects(escaped)
    if len(objects) >= 2:
        merged: dict[str, object] = {}
        for obj in objects:
            for key, value in obj.items():
                if _is_empty_value(merged.get(key)):
                    merged[key] = value
        if _has_command(merged):
            return merged

    if single is not None:
        return single
    raise ValueError("Could not parse a JSON object from the model response.")


_RESERVED_TOP_LEVEL_KEYS = {"thought", "action", "action_input", "scratchpad", "answer", "input"}


def _recover_action_and_input(payload: dict[str, object]) -> tuple[object, object]:
    """Tolerate common key-naming drift from LLMs: ``input`` instead of
    ``action_input``; a bare ``answer``/``value``/``result`` object meaning the
    model wanted to answer; and tool params left at the top level."""
    action = payload.get("action")
    action_input = payload.get("action_input", {})

    if (not action_input) and isinstance(payload.get("input"), dict):
        action_input = payload["input"]

    if action is None and "answer" in payload:
        action = "answer"
        raw_answer = payload["answer"]
        action_input = raw_answer if isinstance(raw_answer, dict) else {"answer": raw_answer}
    elif action is None and any(k in payload for k in ("value", "result", "count", "data")):
        action = "answer"
        action_input = {k: v for k, v in payload.items() if k not in {"thought", "scratchpad"}}

    # Move stray top-level tool params (e.g. {"action":"execute_python","code":...})
    # into action_input without overwriting explicit keys.
    if isinstance(action_input, dict):
        action_input = dict(action_input)
        for key, value in payload.items():
            if key not in _RESERVED_TOP_LEVEL_KEYS and key not in action_input:
                action_input[key] = value

    return action, action_input


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_merged_json_object(normalized)

    thought = payload.get("thought", "")
    action, action_input = _recover_action_and_input(payload)
    scratchpad = payload.get("scratchpad", "")
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")
    if not isinstance(scratchpad, str):
        scratchpad = str(scratchpad)

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
        scratchpad=scratchpad,
    )


# Inline correction for the common qwen failure mode where the model returns a
# valid JSON object with thought/scratchpad but DROPS the mandatory `action`
# field. Instead of burning a step as __error__, we re-prompt in place.
_MAX_ACTION_RETRIES = 2
_MISSING_ACTION_CORRECTION = (
    "Your previous response was REJECTED because it had NO `action` field — it "
    "contained only thought/scratchpad (a plan), not a tool call. You MUST reply "
    "with exactly ONE JSON object that includes a non-empty `action` (the tool to "
    "run RIGHT NOW) plus `action_input`, alongside `thought` and `scratchpad`. "
    "Do NOT reply with planning text only. Pick the concrete next tool and emit it, "
    'for example: {"thought":"inspect files first","scratchpad":"## Plan\\n- list, '
    'then read","action":"list_context","action_input":{}}'
)


# Stall / loop-breaker thresholds (steps).
_STALL_SAME_SIG_WARN = 2        # same failing action+input repeated N times -> switch hint
_STALL_CONSEC_FAIL_NUDGE = 4    # any N failures in a row -> consider answering / re-plan
_STALL_CONSEC_FAIL_BREAK = 12   # runaway loop -> abort to avoid burning all steps
_FORCE_ANSWER_REMAINING = 11    # remaining steps <= N -> remind to answer (step 60 of 70)
_MAX_ANSWER_REJECTIONS = 2      # verifier rejections allowed before accepting (fail-closed budget)
_MAX_PHANTOM_REJECTIONS = 3     # extra bounces allowed when the verifier demands a non-existent table/column
# Anti-thrash (P2): cumulative repeats of the SAME action+input across the whole
# trace, and cumulative count of pure-discovery calls (schema/file listing) that
# make no progress toward an answer.
_STALL_REPEAT_TOTAL = 4         # same signature seen >= N times total -> stop-repeating hint
_DISCOVERY_CALL_LIMIT = 8       # >= N exploration/listing calls total -> commit-now hint


def _step_signature(step: StepRecord) -> str:
    return f"{step.action}:{repr(step.action_input)[:200]}"


# Pure exploration tools: they reveal what exists but never compute an answer.
_DISCOVERY_ACTIONS = frozenset(
    {"list_context", "glob_context", "inspect_sqlite_schema", "profile_column"}
)


def _is_discovery_step(step: StepRecord) -> bool:
    """True for schema/file-listing calls that make no progress toward an answer.

    Includes the dedicated listing tools plus SQL that only enumerates the schema
    (sqlite_master / PRAGMA), which agents otherwise spam to "re-orient"."""
    if step.action in _DISCOVERY_ACTIONS:
        return True
    if step.action == "execute_context_sql":
        sql = str(step.action_input.get("sql", "")).lower()
        if "sqlite_master" in sql or "pragma" in sql:
            return True
    return False


def _stall_hint(steps: list[StepRecord]) -> tuple[str, bool]:
    """Inspect the tail of the trace and return (hint, should_abort).

    Detects two failure loops: (a) the same action+input repeated while failing,
    and (b) many consecutive failed steps regardless of action. Returns a nudge
    for the model and, only for a runaway loop, a request to abort."""
    if not steps:
        return "", False

    consec_fail = 0
    for step in reversed(steps):
        if step.ok:
            break
        consec_fail += 1

    # Non-failing loop: the model keeps re-issuing the SAME call (e.g. re-reading
    # the same file) even though it succeeds, making no progress. Count repeats of
    # the latest signature within a recent window regardless of ok.
    last_sig = _step_signature(steps[-1])
    window = steps[-6:]
    repeat_sig = sum(1 for step in window if _step_signature(step) == last_sig)
    if consec_fail == 0:
        if repeat_sig >= 3:
            return (
                f"You have already run `{steps[-1].action}` with the same input "
                f"{repeat_sig} times and you already have its result. Do NOT repeat it. "
                "Use what you have to make progress or submit your answer now.",
                False,
            )
        # P2a: same successful call repeated too often across the WHOLE task (even
        # if interleaved with other steps, so the window check above misses it).
        total_repeat = sum(1 for step in steps if _step_signature(step) == last_sig)
        if total_repeat >= _STALL_REPEAT_TOTAL:
            return (
                f"You have run `{steps[-1].action}` with the same input {total_repeat} "
                "times across this task and already have its result. Stop re-running it - "
                "query the data you still need, or submit your answer now.",
                False,
            )
        # P2b: too many pure-discovery / listing calls without ever committing - the
        # classic "busy-work" loop where the agent re-orients instead of answering.
        if _is_discovery_step(steps[-1]):
            discovery = sum(1 for step in steps if _is_discovery_step(step))
            if discovery >= _DISCOVERY_CALL_LIMIT:
                return (
                    f"You have made {discovery} schema/file-listing exploration calls. You "
                    "have inspected the data sources MORE than enough. STOP exploring: write "
                    "the SQL/Python that computes the answer from the authoritative table named "
                    "in the knowledge guide, or submit the answer you already have now.",
                    False,
                )
        return "", False

    same_sig = 0
    for step in reversed(steps):
        if _step_signature(step) == last_sig:
            same_sig += 1
        else:
            break

    if consec_fail >= _STALL_CONSEC_FAIL_BREAK:
        return "", True

    if same_sig >= _STALL_SAME_SIG_WARN:
        return (
            f"STOP REPEATING: you have run `{steps[-1].action}` with the same input "
            f"{same_sig} times and it keeps failing. This approach is NOT working - do "
            "NOT try it again. Switch to a COMPLETELY different strategy: inspect the "
            "schema/files first (inspect_sqlite_schema, list_context), query the data a "
            "different way (execute_context_sql can now query CSV/JSON tables too), or "
            "fix the exact error shown. If you already have enough data, answer now.",
            False,
        )

    if consec_fail >= _STALL_CONSEC_FAIL_NUDGE:
        return (
            f"You have hit {consec_fail} errors in a row. The current plan may be flawed. "
            "Re-read the latest error, try a fundamentally different tool/approach, or "
            "submit your answer with the information you already have.",
            False,
        )

    return "", False


# Tools whose inputs reveal WHICH data source produced an answer. Scanned at
# submission time to give the verifier provenance (decoy file vs authoritative table).
_PROVENANCE_ACTIONS = {
    "read_csv", "read_json", "read_pdf", "read_doc", "extract_pdf_tables", "extract_records",
    "execute_context_sql", "inspect_sqlite_schema", "execute_python", "submit_csv",
}
_SQL_TABLE_RE = re.compile(r"(?:from|join)\s+[\"`\[]?([A-Za-z_][\w]*)", re.IGNORECASE)
_PY_FILE_RE = re.compile(r"""['\"]([\w./\-]+\.(?:csv|json|sqlite|db|pdf))['\"]""")


def _answer_provenance(steps: list[StepRecord], limit: int = 8) -> list[str]:
    """Extract the file paths / SQL tables touched by recent SUCCESSFUL data steps.

    The most recent data step that fed the answer is the strongest signal for
    whether the agent pulled from a top-level decoy file or the authoritative
    table named in the knowledge guide."""
    refs: list[str] = []
    for rec in reversed(steps):
        if len(refs) >= limit:
            break
        if not rec.ok or rec.action not in _PROVENANCE_ACTIONS:
            continue
        ai = rec.action_input or {}
        act = rec.action
        if act in {"read_csv", "read_json", "read_pdf", "read_doc", "extract_pdf_tables", "extract_records"}:
            path = ai.get("path")
            if path:
                refs.append(f"{act}({path})")
        elif act in {"execute_context_sql", "inspect_sqlite_schema"}:
            sql = str(ai.get("sql") or ai.get("query") or "")
            tables = list(dict.fromkeys(_SQL_TABLE_RE.findall(sql)))
            if tables:
                refs.append(f"sql FROM {', '.join(tables)}")
            elif ai.get("path"):
                refs.append(f"{act}({ai.get('path')})")
        elif act == "execute_python":
            code = str(ai.get("code", ""))
            files = [f for f in dict.fromkeys(_PY_FILE_RE.findall(code)) if f != "answer.csv"]
            tables = list(dict.fromkeys(_SQL_TABLE_RE.findall(code)))
            hits = files + [f"table:{t}" for t in tables if t.lower() != "df"]
            if hits:
                refs.append("execute_python reads " + ", ".join(hits[:6]))
    return refs


_CREATE_COL_RE = re.compile(
    r'"(\w+)"\s+(?:integer|int|text|real|numeric|blob|date|datetime|time|'
    r'float|double|decimal|varchar|char|bool|boolean)',
    re.IGNORECASE,
)


def _observed_schema(steps: list[StepRecord], *, max_chars: int = 3500) -> str:
    """Build a compact REACHABLE map of tables/files+columns the agent has actually
    observed in THIS task (from successful schema inspections, csv reads, and SQL
    results). The verifier uses this to avoid demanding an "authoritative" table or
    column that only exists in the knowledge guide but is NOT physically present:
    when the canonical source is unreachable, the closest reachable column should be
    accepted instead of bouncing the agent into a dead end."""
    tables: dict[str, list[str]] = {}
    files: dict[str, list[str]] = {}
    bare_files: list[str] = []

    def _add(store: dict[str, list[str]], name: str, cols: list[str]) -> None:
        if not name:
            return
        existing = store.setdefault(name, [])
        for c in cols:
            c = str(c).strip()
            if c and c not in existing and len(existing) < 30:
                existing.append(c)

    for rec in steps:
        if not rec.ok:
            continue
        obs = rec.observation if isinstance(rec.observation, dict) else {}
        content = obs.get("content") if isinstance(obs, dict) else None
        if not isinstance(content, dict):
            continue
        act = rec.action
        if act == "inspect_sqlite_schema":
            for tbl in content.get("tables", []) or []:
                if not isinstance(tbl, dict):
                    continue
                name = str(tbl.get("name", "")).strip()
                cols = _CREATE_COL_RE.findall(str(tbl.get("create_sql", "")))
                _add(tables, name, cols)
        elif act in {"read_csv", "read_json"}:
            path = str((rec.action_input or {}).get("path", "")).strip()
            cols = content.get("columns") if isinstance(content.get("columns"), list) else []
            if path:
                _add(files, path, [str(c) for c in cols])
        elif act == "execute_context_sql":
            sql = str((rec.action_input or {}).get("sql") or "")
            cols = content.get("columns") if isinstance(content.get("columns"), list) else []
            for t in dict.fromkeys(_SQL_TABLE_RE.findall(sql)):
                _add(tables, t, [str(c) for c in cols])
        elif act in {"list_context", "glob_context"}:
            for entry in content.get("entries", []) or []:
                if isinstance(entry, dict) and entry.get("kind") == "file":
                    bare_files.append(str(entry.get("path", "")))
            for m in content.get("matches", []) or []:
                bare_files.append(str(m))

    parts: list[str] = []
    if tables:
        parts.append(
            "DB tables present (with observed columns): "
            + "; ".join(
                f"{name}({', '.join(cols)})" if cols else name
                for name, cols in tables.items()
            )
        )
    if files:
        parts.append(
            "Files read (with columns): "
            + "; ".join(
                f"{name}[{', '.join(cols)}]" if cols else name
                for name, cols in files.items()
            )
        )
    if bare_files:
        seen = list(dict.fromkeys(f for f in bare_files if f))
        parts.append("Other files listed: " + ", ".join(seen[:40]))

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(truncated)"
    return text


_PHANTOM_TABLE_RE = re.compile(r"no such table:\s*([A-Za-z_][\w]*)", re.IGNORECASE)
_MISSING_ASSET_RE = re.compile(r"Missing context asset:\s*([\w./\-]+)", re.IGNORECASE)


def _phantom_tables(steps: list[StepRecord]) -> set[str]:
    """Names the agent tried to query/open but that do NOT exist (SQLite 'no such
    table' errors and 'Missing context asset' errors). Used to stop the verifier
    from repeatedly demanding a knowledge-guide table that is physically absent."""
    phantom: set[str] = set()
    for rec in steps:
        obs = rec.observation if isinstance(rec.observation, dict) else {}
        err = str(obs.get("error", "")) if isinstance(obs, dict) else ""
        if not err:
            continue
        for m in _PHANTOM_TABLE_RE.findall(err):
            phantom.add(m)
        for m in _MISSING_ASSET_RE.findall(err):
            # keep the bare stem (e.g. db/qt_dailyquote.sqlite -> qt_dailyquote)
            stem = m.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if stem:
                phantom.add(stem)
    return phantom


_MISSING_SOURCE_SIGNALS = (
    "no such table",
    "missing context asset",
    "file is not a database",
)


def _table_search_thrashing(
    steps: list[StepRecord], *, window: int = 10, threshold: int = 3
) -> bool:
    """True when the agent is spinning over the data sources without progress: it
    either repeatedly hits 'the source does not exist' (no such table / missing
    asset / not a database) OR keeps re-reading the very same file/path. Both mean
    it's stuck hunting structured data that isn't paying off - the cue to fall back
    on already-extracted video/document evidence."""
    recent = steps[-window:]
    miss_hits = 0
    path_counts: dict[str, int] = {}
    for rec in recent:
        obs = rec.observation if isinstance(rec.observation, dict) else {}
        err = str(obs.get("error", "")).lower() if isinstance(obs, dict) else ""
        if err and any(sig in err for sig in _MISSING_SOURCE_SIGNALS):
            miss_hits += 1
        path = ""
        if isinstance(rec.action_input, dict):
            path = str(rec.action_input.get("path") or "").strip().lower()
        if path:
            path_counts[path] = path_counts.get(path, 0) + 1
    repeat_hits = max(path_counts.values(), default=0)
    return miss_hits >= threshold or repeat_hits >= threshold


def _reachable_identifiers(
    steps: list[StepRecord],
) -> tuple[set[str], set[str], set[str]]:
    """Return (table_names, column_names, file_names) the agent has actually observed
    in this task, all lower-cased. Used to detect when the verifier demands a
    table/column that does NOT physically exist so the phantom demand can be
    neutralised instead of looping the agent to death."""
    tables: set[str] = set()
    columns: set[str] = set()
    files: set[str] = set()
    for rec in steps:
        if not rec.ok:
            continue
        obs = rec.observation if isinstance(rec.observation, dict) else {}
        content = obs.get("content") if isinstance(obs, dict) else None
        if not isinstance(content, dict):
            continue
        act = rec.action
        if act == "inspect_sqlite_schema":
            for tbl in content.get("tables", []) or []:
                if isinstance(tbl, dict):
                    tables.add(str(tbl.get("name", "")).strip().lower())
                    for c in _CREATE_COL_RE.findall(str(tbl.get("create_sql", ""))):
                        columns.add(c.lower())
        elif act in {"read_csv", "read_json"}:
            for c in content.get("columns", []) or []:
                columns.add(str(c).strip().lower())
            path = str((rec.action_input or {}).get("path", "")).strip()
            if path:
                files.add(path.rsplit("/", 1)[-1].lower())
        elif act == "execute_context_sql":
            for c in content.get("columns", []) or []:
                columns.add(str(c).strip().lower())
            for t in _SQL_TABLE_RE.findall(str((rec.action_input or {}).get("sql") or "")):
                tables.add(t.lower())
        elif act in {"list_context", "glob_context"}:
            for entry in content.get("entries", []) or []:
                if isinstance(entry, dict):
                    p = str(entry.get("path", ""))
                    if p:
                        files.add(p.rsplit("/", 1)[-1].lower())
            for m in content.get("matches", []) or []:
                files.add(str(m).rsplit("/", 1)[-1].lower())
    tables.discard("")
    columns.discard("")
    files.discard("")
    return tables, columns, files


def _unreachable_demands(feedback: str, steps: list[StepRecord]) -> list[str]:
    """Backtick-quoted identifiers in the verifier feedback that name a table/column
    which does NOT exist in the observed data (a phantom demand). Includes any names
    the agent already hit 'no such table' / 'missing asset' on."""
    tables, columns, files = _reachable_identifiers(steps)
    phantom = {p.lower() for p in _phantom_tables(steps)}
    out: list[str] = []
    seen: set[str] = set()
    # The verifier quotes identifiers inconsistently: backticks, single or double
    # quotes. Pull tokens from all three styles.
    candidates = re.findall(r"[`'\"]([^`'\"]+)[`'\"]", feedback or "")
    for tok in candidates:
        raw = tok.strip()
        low = raw.lower()
        if low in seen:
            continue
        # Only schema-identifier-looking tokens: start with a letter, contain only
        # [a-z0-9_], and either have an underscore or be reasonably long (avoids
        # flagging numeric values like 601908 or short English words).
        if not re.fullmatch(r"[a-z][a-z0-9_]*", low):
            continue
        if "_" not in low and len(low) < 8:
            continue
        if low in phantom:
            out.append(raw)
            seen.add(low)
            continue
        if low in tables or low in columns:
            continue
        if any(low == f or low == f.rsplit(".", 1)[0] for f in files):
            continue
        out.append(raw)
        seen.add(low)
    return out


def _classify_loose_sources(context_dir: Path) -> list[str]:
    """List top-level CSV/JSON files (directly in the context root, not in a
    csv/ json/ db/ doc/ subdir). These loose files are frequently pre-computed
    answers/decoys (`*_result.csv`, `*_sorted.csv`, `top_100_*.json`) whose row
    count can match gold while values are corrupted, mis-columned, or stale."""
    loose: list[str] = []
    try:
        for path in sorted(context_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
                loose.append(path.name)
    except OSError:
        return []
    return loose


def _collect_result_tables(
    steps: list[StepRecord], limit: int = 6, max_full_rows: int = 5000, preview: int = 12
) -> list[dict[str, Any]]:
    """Gather tabular results the agent already computed (successful SQL outputs and
    any tool returning columns+rows) so a forced final answer can reuse exact rows.
    Returns most-recent-first blocks with both full rows (for materialization) and a
    small preview (for the LLM prompt)."""
    tables: list[dict[str, Any]] = []
    for rec in reversed(steps):
        if len(tables) >= limit:
            break
        if not rec.ok:
            continue
        obs = rec.observation if isinstance(rec.observation, dict) else {}
        content = obs.get("content") if isinstance(obs, dict) else None
        if not isinstance(content, dict):
            continue
        cols = content.get("columns")
        rows = content.get("rows")
        if not (isinstance(cols, list) and cols and isinstance(rows, list) and rows):
            continue
        str_cols = [str(c) for c in cols]
        full_rows = [list(r) for r in rows[:max_full_rows]]
        tables.append(
            {
                "step_index": rec.step_index,
                "columns": str_cols,
                "rows_full": full_rows,
                "preview_rows": full_rows[:preview],
                "row_count": len(rows),
            }
        )
    for i, tbl in enumerate(tables):
        tbl["index"] = i
    return tables


def _collect_python_outputs(
    steps: list[StepRecord], limit: int = 4, max_chars: int = 1500
) -> list[str]:
    """Recent successful execute_python stdout snippets (most-recent-first)."""
    out: list[str] = []
    for rec in reversed(steps):
        if len(out) >= limit:
            break
        if rec.action != "execute_python" or not rec.ok:
            continue
        obs = rec.observation if isinstance(rec.observation, dict) else {}
        content = obs.get("content") if isinstance(obs, dict) else {}
        text = ""
        if isinstance(content, dict):
            text = str(content.get("output") or "")
        if text.strip():
            out.append(f"[python step {rec.step_index} stdout]\n{text[:max_chars]}")
    return out


def _materialize_forced_answer(
    spec: dict[str, Any], result_tables: list[dict[str, Any]]
) -> AnswerTable | None:
    """Turn the force-answer agent's JSON spec into an AnswerTable. When it points at
    a captured result table, reuse that table's FULL rows (optionally projecting to
    the requested column subset) so multi-row values are exact rather than retyped."""
    if not isinstance(spec, dict):
        return None
    idx = spec.get("use_result_index")
    if idx is not None:
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = None
        if idx is not None and 0 <= idx < len(result_tables):
            tbl = result_tables[idx]
            cols = tbl["columns"]
            rows = tbl["rows_full"]
            keep = spec.get("columns")
            if isinstance(keep, list) and keep:
                indices = [cols.index(str(c)) for c in keep if str(c) in cols]
                if indices:
                    new_cols = [cols[i] for i in indices]
                    new_rows = [
                        [r[i] if i < len(r) else "" for i in indices] for r in rows
                    ]
                    return AnswerTable(columns=new_cols, rows=new_rows)
            return AnswerTable(columns=list(cols), rows=[list(r) for r in rows])
    cols = spec.get("columns")
    rows = spec.get("rows")
    if isinstance(cols, list) and cols and isinstance(rows, list):
        return AnswerTable(
            columns=[str(c) for c in cols], rows=[list(r) for r in rows]
        )
    return None


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=_build_initial_user_content(task)))

        if state.scratchpad.strip():
            messages.append(
                ModelMessage(
                    role="user",
                    content=(
                        "## Working Memory (scratchpad)\n"
                        "These are your durable notes for this task. Older step details below "
                        "may be summarized, so rely on this section for facts that must persist.\n\n"
                        f"{_truncate_text(state.scratchpad.strip(), _SCRATCHPAD_CHAR_CAP)}"
                    ),
                )
            )

        if state.video_evidence:
            messages.append(
                ModelMessage(
                    role="user",
                    content=(
                        "## Video Evidence (auto-recorded, never dropped)\n"
                        "Verified readings from inspect_video. These values are AUTHORITATIVE: "
                        "use them as filter thresholds AND - critically - they may BE the "
                        "answer the question asks for (e.g. a per-group max/total already shown "
                        "on a slide). If the knowledge guide names a table/column that you "
                        "cannot find in this task's files, do NOT keep hunting for it - the "
                        "requested numbers are likely already here; answer directly from these "
                        "readings (heed any 'this page is average/category-peak, not the asked "
                        "metric' caveats and pick the page that matches the question). Re-run "
                        "inspect_video on a narrow window only if a needed value is missing or "
                        "low-confidence. When the answer is a per-group value, return one "
                        "row for EVERY group present in the evidence - do NOT drop any "
                        "category (e.g. keep 博士后/postdoc or other minor groups); omitting "
                        "even one row makes the whole column mismatch and score zero.\n\n"
                        f"{_render_video_evidence(state.video_evidence)}"
                    ),
                )
            )

        total = len(state.steps)
        for idx, step in enumerate(state.steps):
            is_recent = idx >= total - self.config.recent_full_steps
            if is_recent:
                messages.append(ModelMessage(role="assistant", content=step.raw_response))
                obs_text = build_observation_prompt(step.observation)
                messages.append(
                    ModelMessage(role="user", content=_truncate_text(obs_text, _RECENT_OBS_CHAR_CAP))
                )
            else:
                messages.append(ModelMessage(role="user", content=_summarize_step(step)))

        if state.pending_hint.strip():
            messages.append(ModelMessage(role="user", content=state.pending_hint.strip()))
        return messages

    def _load_knowledge_docs(self, task: PublicTask) -> str:
        """Read the task's knowledge guide(s) so the planner can map business
        terms to the exact source table/column. These tasks ship a semantic data
        dictionary (knowledge.md / KNOWLEDGE_GUIDE.md / doc/*.md); grounding the
        plan in it avoids picking a same-named-but-wrong column from whichever
        table happens to be convenient (e.g. SQLite vs CSV)."""
        context_dir = getattr(task, "context_dir", None)
        if context_dir is None:
            return ""
        context_dir = Path(context_dir)
        seen: set[Path] = set()
        candidates: list[Path] = []
        # Priority: top-level knowledge files, then any markdown under doc/.
        for pattern in ("knowledge*.md", "KNOWLEDGE*.md", "*.md", "doc/**/*.md"):
            for path in sorted(context_dir.glob(pattern)):
                resolved = path.resolve()
                if path.is_file() and resolved not in seen:
                    seen.add(resolved)
                    candidates.append(path)
        if not candidates:
            return ""
        budget = 14000
        chunks: list[str] = []
        for path in candidates:
            if budget <= 0:
                break
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = path.relative_to(context_dir)
            snippet = text[:budget]
            chunks.append(f"### {rel}\n{snippet}")
            budget -= len(snippet)
        if not chunks:
            return ""
        return "## Knowledge guide (authoritative term -> table/column mapping)\n" + "\n\n".join(chunks)

    def _build_table_source_index(self, task: PublicTask) -> str:
        """Map each logical table name to its concrete SOURCE (a CSV file vs a
        table inside a SQLite db). Tables with the SAME name can exist in both
        (e.g. an ``AFloats`` column appears in both ``lc_sharestru`` in SQLite and
        ``lc_freefloat`` as a CSV); making the source explicit lets the planner
        pick the table the knowledge guide actually points to instead of whichever
        is most convenient to query."""
        import json as _json  # noqa: PLC0415
        import sqlite3 as _sqlite3  # noqa: PLC0415

        context_dir = getattr(task, "context_dir", None)
        if context_dir is None:
            return ""
        context_dir = Path(context_dir)
        lines: list[str] = []

        for csv_path in sorted(context_dir.rglob("*.csv")):
            rel = csv_path.relative_to(context_dir).as_posix()
            try:
                with csv_path.open(encoding="utf-8", errors="replace") as handle:
                    n_rows = max(0, sum(1 for _ in handle) - 1)
                count = f" ({n_rows} data rows)"
            except OSError:
                count = ""
            decoy = " ⚠ TOP-LEVEL LOOSE FILE - likely a pre-computed DECOY; do NOT use as the answer source, prefer the authoritative knowledge-guide table" if "/" not in rel else ""
            lines.append(
                f"- `{csv_path.stem}` -> CSV file `{rel}`{count} (load with pd.read_csv){decoy}"
            )

        for json_path in sorted(context_dir.rglob("*.json")):
            rel = json_path.relative_to(context_dir).as_posix()
            try:
                payload = _json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                continue
            records = None
            table_name = json_path.stem
            if isinstance(payload, dict):
                if isinstance(payload.get("records"), list):
                    records = payload["records"]
                    table_name = str(payload.get("table", table_name))
                else:
                    for value in payload.values():
                        if isinstance(value, list):
                            records = value
                            break
            elif isinstance(payload, list):
                records = payload
            if not isinstance(records, list):
                continue
            cols = ""
            if records and isinstance(records[0], dict):
                cols = ", ".join(str(k) for k in records[0].keys())
            decoy = " ⚠ TOP-LEVEL LOOSE FILE - likely a pre-computed DECOY; prefer the authoritative knowledge-guide table" if "/" not in rel else ""
            lines.append(
                f"- `{table_name}` -> JSON file `{rel}` ({len(records)} records; "
                f"columns: {cols}) (load with json.load then pd.DataFrame(data['records'])){decoy}"
            )

        for pattern in ("*.sqlite", "*.db", "*.sqlite3"):
            for db_path in sorted(context_dir.rglob(pattern)):
                rel = db_path.relative_to(context_dir).as_posix()
                try:
                    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
                    con = _sqlite3.connect(uri, uri=True)
                    names = [
                        row[0]
                        for row in con.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' "
                            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                        ).fetchall()
                    ]
                    con.close()
                except _sqlite3.Error:
                    continue
                for name in names:
                    lines.append(
                        f"- `{name}` -> table inside SQLite db `{rel}` "
                        f"(query with execute_context_sql, or pd.read_sql)"
                    )

        if not lines:
            return ""
        return (
            "## Table source index (where each table actually lives)\n"
            "Use the knowledge guide to pick WHICH table a business concept maps to, "
            "then use this index to load it from the correct source. If a name appears "
            "as both a CSV and a SQLite table, they are DIFFERENT tables - follow the "
            "knowledge guide's definition. Files marked ⚠ TOP-LEVEL LOOSE FILE are "
            "pre-computed decoys (their row count may match the answer but values are "
            "wrong/mis-columned/stale) - NEVER answer from them; always go to the "
            "authoritative table in csv/ json/ db/ doc/.\n" + "\n".join(lines)
        )

    async def _build_planner_context(self, task: PublicTask) -> str:
        """Gather cheap context for the one-shot planner."""
        parts: list[str] = []

        knowledge = await asyncio.to_thread(self._load_knowledge_docs, task)
        if knowledge:
            parts.append(knowledge)

        source_index = await asyncio.to_thread(self._build_table_source_index, task)
        if source_index:
            parts.append(source_index)

        try:
            list_result = await asyncio.to_thread(self.tools.execute, task, "list_context", {"max_depth": 4})
            parts.append("## Context files\n" + json.dumps(list_result.content, ensure_ascii=False)[:12000])
        except Exception as exc:  # noqa: BLE001
            parts.append(f"## Context files\nFailed to list context: {exc}")

        db_paths: list[str] = []
        for pattern in ("**/*.sqlite", "**/*.db"):
            try:
                glob_result = await asyncio.to_thread(
                    self.tools.execute,
                    task,
                    "glob_context",
                    {"pattern": pattern},
                )
                matches = glob_result.content.get("matches", []) if isinstance(glob_result.content, dict) else []
                db_paths.extend(str(item) for item in matches)
            except Exception:
                continue
        db_paths = list(dict.fromkeys(db_paths))[:2]

        for db_path in db_paths:
            try:
                schema_result = await asyncio.to_thread(
                    self.tools.execute,
                    task,
                    "inspect_sqlite_schema",
                    {"path": db_path},
                )
                parts.append(
                    f"## SQLite schema: {db_path}\n"
                    + json.dumps(schema_result.content, ensure_ascii=False)[:12000]
                )
            except Exception as exc:  # noqa: BLE001
                parts.append(f"## SQLite schema: {db_path}\nFailed to inspect schema: {exc}")

        return "\n\n".join(parts)

    async def _prime_verifier_context(self, task: PublicTask, state: AgentRuntimeState) -> None:
        """Cache the knowledge guide + loose decoy list once, for the answer-time
        source/column alignment check (independent of whether the planner runs)."""
        try:
            state.knowledge_text = await asyncio.to_thread(self._load_knowledge_docs, task)
        except Exception:  # noqa: BLE001
            state.knowledge_text = ""
        context_dir = getattr(task, "context_dir", None)
        if context_dir is not None:
            try:
                state.decoy_files = await asyncio.to_thread(
                    _classify_loose_sources, Path(context_dir)
                )
            except Exception:  # noqa: BLE001
                state.decoy_files = []

    async def _initialize_scratchpad(self, task: PublicTask, state: AgentRuntimeState) -> None:
        if not self.config.enable_planner:
            return
        try:
            context_summary = await self._build_planner_context(task)
            plan = await run_planner(
                model=self.model,
                question=task.question,
                context_summary=context_summary,
            )
            if plan:
                state.scratchpad = f"## Initial plan\n{plan}"
        except Exception as exc:  # noqa: BLE001
            state.scratchpad = (
                "## Initial plan\n"
                f"Planner failed ({exc}). Proceed with standard ReAct exploration."
            )

    async def run(self, task: PublicTask) -> AgentRunResult:
        try:
            return await self._run_loop(task)
        finally:
            close = getattr(self.tools, "close", None)
            if callable(close):
                close()

    async def _run_loop(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        raw_response = ""
        await self._prime_verifier_context(task, state)
        await self._initialize_scratchpad(task, state)
        for step_index in range(1, self.config.max_steps + 1):
            hint, should_abort = _stall_hint(state.steps)
            if should_abort:
                state.failure_reason = (
                    "Aborted after too many consecutive failing steps (runaway loop)."
                )
                break
            # When the agent is thrashing on a non-existent data source but already
            # has video evidence in hand, redirect it to answer from that evidence
            # instead of burning the whole budget chasing a phantom table.
            if state.video_evidence and _table_search_thrashing(state.steps):
                redirect = (
                    "REDIRECT: you have repeatedly tried to open a data source that does "
                    "NOT exist in this task. STOP searching for tables/files named by the "
                    "knowledge guide. The values the question needs are already in the "
                    "Video Evidence above - select the ones that match the asked metric "
                    "(mind any 'average/category-peak, not the asked total' caveats) and "
                    "call `answer` NOW with ONLY the requested column(s). Emit one row for "
                    "EVERY group/category listed in the evidence - do NOT drop any group "
                    "(e.g. include 博士后/postdoc or any minor category); a missing row "
                    "makes the whole column mismatch and scores zero."
                )
                hint = f"{hint}\n\n{redirect}" if hint else redirect
            remaining = self.config.max_steps - step_index + 1
            if remaining <= _FORCE_ANSWER_REMAINING:
                if remaining <= 3:
                    urgency = (
                        f"FINAL WARNING: only {remaining} step(s) left out of "
                        f"{self.config.max_steps}. You MUST call `answer` (or `submit_csv` "
                        "for many rows) on THIS step with the best result you have - an "
                        "imperfect answer scores more than no answer. Do NOT run any more "
                        "exploration. Return ONLY the column(s) the question asks for."
                    )
                else:
                    urgency = (
                        f"You are in the final phase: {remaining} of {self.config.max_steps} "
                        "steps remain. Start finalizing NOW - decide your answer directly. "
                        "Run at most one more essential check, then call `answer` (or "
                        "`submit_csv` for many rows) with ONLY the requested column(s). Do "
                        "not start new exploration paths."
                    )
                hint = f"{hint}\n\n{urgency}" if hint else urgency
            state.pending_hint = hint
            messages = self._build_messages(task, state)
            raw_response = await self.model.complete(messages)
            # Inline correction: if the model dropped the mandatory `action` field
            # (qwen often emits thought/scratchpad-only plans), re-prompt in place
            # with a forceful reminder rather than wasting the step as __error__.
            for _ in range(_MAX_ACTION_RETRIES):
                try:
                    parse_model_step(raw_response)
                    break
                except ValueError as parse_exc:
                    if "action must be a non-empty" not in str(parse_exc):
                        break
                    raw_response = await self.model.complete(
                        messages
                        + [
                            ModelMessage(role="assistant", content=raw_response),
                            ModelMessage(role="user", content=_MISSING_ACTION_CORRECTION),
                        ]
                    )
            try:
                model_step = parse_model_step(raw_response)
                # 每轮响应都携带 scratchpad 字段：随动作一起更新工作记忆，零额外轮次。
                if model_step.scratchpad.strip():
                    state.scratchpad = model_step.scratchpad.strip()

                if model_step.action == "memory_grep":
                    raw_step_idx = model_step.action_input.get("step_index")
                    try:
                        step_idx = int(raw_step_idx) if raw_step_idx is not None else None
                    except (TypeError, ValueError):
                        step_idx = None
                    try:
                        max_matches = int(model_step.action_input.get("max_matches", 5))
                    except (TypeError, ValueError):
                        max_matches = 5
                    grep_content = _memory_grep(
                        state.steps,
                        pattern=(str(model_step.action_input.get("pattern")).strip()
                                 if model_step.action_input.get("pattern") else None),
                        step_index=step_idx,
                        max_matches=max(1, max_matches),
                    )
                    observation = {
                        "ok": grep_content.get("status") == "ok",
                        "tool": "memory_grep",
                        "content": grep_content,
                    }
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation=observation,
                            ok=bool(observation["ok"]),
                        )
                    )
                    continue

                # 工具执行是 CPU/IO 密集型，放到线程池避免阻塞事件循环
                tool_result = await asyncio.to_thread(
                    self.tools.execute, task, model_step.action, model_step.action_input
                )
                content = dict(tool_result.content)
                # Auto-archive successful video readings into durable structured memory
                # so verified thresholds are never lost to the sliding-window summarizer.
                if model_step.action == "inspect_video" and tool_result.ok:
                    state.video_evidence.append(
                        {
                            "step": step_index,
                            "window": content.get("window"),
                            "query": model_step.action_input.get("query"),
                            "confidence": content.get("confidence"),
                            "observation": content.get("observation", ""),
                        }
                    )
                # Answer-time source/column alignment check. Runs once, for BOTH
                # `answer` (inline rows) and `submit_csv` (rows read from a file),
                # so large results don't bypass verification. Rejection bounces the
                # answer back with the decoy/column feedback instead of terminating.
                if (
                    tool_result.is_terminal
                    and tool_result.answer is not None
                    and self.config.enable_answer_verify
                    and state.answer_rejections < _MAX_ANSWER_REJECTIONS
                    and state.phantom_rejections < _MAX_PHANTOM_REJECTIONS
                ):
                    ans = tool_result.answer
                    provenance = _answer_provenance(state.steps)
                    prov_sig = "; ".join(provenance) if provenance else "(unknown)"
                    verdict = await verify_answer(
                        model=self.model,
                        question=task.question,
                        scratchpad=state.scratchpad,
                        columns=[str(col) for col in ans.columns],
                        rows=[list(row) for row in ans.rows],
                        knowledge=state.knowledge_text,
                        provenance=prov_sig,
                        loose_sources=state.decoy_files,
                        reachable=_observed_schema(state.steps),
                    )
                    content["answer_verifier"] = verdict
                    if verdict.get("decision") == "reject":
                        # Deterministic phantom-source guard: if the verifier is
                        # demanding a table/column that does NOT exist in the observed
                        # data (a knowledge-guide canonical name absent from the real
                        # schema, or one the agent already hit 'no such table' on), the
                        # demand is unsatisfiable. Rewrite the feedback with the REAL
                        # reachable sources/columns so the agent stops chasing the
                        # phantom and picks the closest physical column by meaning. Such
                        # bounces use a SEPARATE budget so they do not burn the real
                        # rejection budget (which would auto-accept a wrong answer).
                        unreachable = _unreachable_demands(
                            f"{verdict.get('feedback', '')} {verdict.get('reason', '')}",
                            state.steps,
                        )
                        phantom_note = ""
                        if unreachable:
                            reachable_summary = _observed_schema(state.steps)
                            phantom_note = (
                                " CRITICAL OVERRIDE: the canonical name(s) "
                                f"{', '.join(unreachable)} do NOT exist as a literal "
                                "table/column here (already confirmed) - do NOT query them "
                                "again, and do NOT give up or answer 'unavailable'/empty. "
                                "The canonical name is the AUTHORITATIVE mapping for the "
                                "question even if it seems counter-intuitive: find the "
                                "REACHABLE column that is its PHYSICAL EQUIVALENT by "
                                "translating the canonical name to a real column name "
                                "(e.g. 'turnoverdeals' = a deals/笔数 column such as 交易笔数, "
                                "NOT a share-volume 成交量 column; 'turnovervolume' = 成交量; "
                                "a daily-growth canonical = the daily-growth column). "
                                f"Sources observed so far:\n{reachable_summary}\nIf the "
                                "equivalent column is not among these, READ the remaining "
                                "candidate files/tables you have not opened yet, then "
                                "submit using that physically-present equivalent column."
                            )
                            state.phantom_rejections += 1
                        else:
                            state.answer_rejections += 1
                        repeat_source = prov_sig in state.rejected_provenance
                        state.rejected_provenance.add(prov_sig)
                        repeat_note = (
                            " You ALREADY submitted an answer from this SAME source and it was "
                            "rejected once - do NOT submit from this source/column again. Switch to "
                            "the authoritative table/column named in the feedback above."
                            if repeat_source else ""
                        )
                        state.steps.append(
                            StepRecord(
                                step_index=step_index,
                                thought=model_step.thought,
                                action=model_step.action,
                                action_input=model_step.action_input,
                                raw_response=raw_response,
                                observation={
                                    "ok": False,
                                    "tool": "answer_verifier",
                                    "content": {
                                        "status": "rejected",
                                        "reason": verdict.get("reason", ""),
                                        "feedback": verdict.get("feedback", ""),
                                        "instruction": (
                                            "Your answer was NOT accepted. Re-derive it from the "
                                            "table/column named in the feedback above - which MUST be a "
                                            "source that actually exists in this task's data (do NOT "
                                            "chase a knowledge-guide table name that you have already "
                                            "confirmed is absent; use its closest real column instead, "
                                            "and avoid loose top-level *_result/*_sorted decoys). Select "
                                            "the exact column(s) the question asks for, drop any extra "
                                            "id/date/index columns, refresh your `scratchpad`, then "
                                            "submit again." + repeat_note + phantom_note
                                        ),
                                    },
                                },
                                ok=False,
                            )
                        )
                        continue

                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

        if state.answer is None:
            # Budget exhausted (or aborted) without a submission. Rather than scoring
            # a guaranteed 0, force a final-answer sub-agent to synthesize the best
            # answer from the result tables / python output already gathered. This
            # path intentionally BYPASSES the verifier - any answer beats no answer.
            result_tables = _collect_result_tables(state.steps)
            python_outputs = _collect_python_outputs(state.steps)
            if result_tables or python_outputs or state.scratchpad.strip():
                try:
                    spec = await force_final_answer(
                        model=self.model,
                        question=task.question,
                        scratchpad=state.scratchpad,
                        knowledge=state.knowledge_text,
                        result_tables=result_tables,
                        python_outputs=python_outputs,
                        video_evidence=(
                            _render_video_evidence(state.video_evidence)
                            if state.video_evidence
                            else ""
                        ),
                    )
                    forced = _materialize_forced_answer(spec, result_tables) if spec else None
                except Exception:  # noqa: BLE001
                    forced = None
                if forced is not None and forced.columns:
                    state.answer = forced
                    state.failure_reason = "forced_final_answer (budget exhausted)"
            if state.answer is None and state.failure_reason is None:
                state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
