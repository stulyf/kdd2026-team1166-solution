from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.filesystem import (
    glob_context,
    grep_context,
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    extract_pdf_tables,
    read_pdf_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.math_calc import calculate as math_calculate
from data_agent_baseline.tools.python_exec import PersistentPythonSession
from data_agent_baseline.tools.sqlite import (
    augment_sqlite_db,
    profile_column,
    execute_read_only_sql,
    inspect_sqlite_schema,
)
from data_agent_baseline.tools.extract import RecordExtractor
from data_agent_baseline.tools.video import VideoInspector

EXECUTE_PYTHON_TIMEOUT_SECONDS = 60
DEFAULT_SQL_LIMIT = 2000


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 50))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 10000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    offset = int(action_input.get("offset", 1))
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=read_doc_preview(task, path, offset=offset, limit=limit))


def _read_pdf(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    start_page = int(action_input.get("start_page", 1))
    max_chars = int(action_input.get("max_chars", 12000))
    content = read_pdf_preview(task, path, start_page=start_page, max_chars=max_chars)
    return ToolExecutionResult(ok="error" not in content, content=content)


def _extract_pdf_tables(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    path = str(action_input["path"])
    start_page = int(action_input.get("start_page", 1))
    max_pages = int(action_input.get("max_pages", 8))
    content = extract_pdf_tables(task, path, start_page=start_page, max_pages=max_pages)
    return ToolExecutionResult(ok="error" not in content, content=content)


def _glob_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    pattern = str(action_input["pattern"])
    path = action_input.get("path")
    return ToolExecutionResult(
        ok=True,
        content=glob_context(task, pattern, path=str(path) if path else None),
    )


def _grep_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    pattern = str(action_input["pattern"])
    path = action_input.get("path")
    include = action_input.get("include")
    context_lines = int(action_input.get("context_lines", 2))
    max_matches = int(action_input.get("max_matches", 50))
    return ToolExecutionResult(
        ok=True,
        content=grep_context(
            task,
            pattern,
            path=str(path) if path else None,
            include=str(include) if include else None,
            context_lines=context_lines,
            max_matches=max_matches,
        ),
    )


def _augmented_db_path(task: PublicTask, path: Path) -> Path:
    """Return an augmented copy of the db (context CSV/JSON loaded as tables),
    falling back to the original db if augmentation fails for any reason."""
    try:
        return augment_sqlite_db(task.context_dir, path)
    except Exception:  # noqa: BLE001 - never let augmentation break a real query
        return path


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    rel_path = str(action_input["path"])
    path = resolve_context_path(task, rel_path)
    content = inspect_sqlite_schema(_augmented_db_path(task, path))
    content["path"] = rel_path  # hide the temp augmented path from the model
    return ToolExecutionResult(ok=True, content=content)


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    rel_path = str(action_input["path"])
    path = resolve_context_path(task, rel_path)
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", DEFAULT_SQL_LIMIT))
    content = execute_read_only_sql(_augmented_db_path(task, path), sql, limit=limit)
    content["path"] = rel_path
    return ToolExecutionResult(ok=True, content=content)


def _profile_column(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    rel_path = str(action_input["path"])
    path = resolve_context_path(task, rel_path)
    table = str(action_input["table"])
    column = str(action_input["column"])
    content = profile_column(_augmented_db_path(task, path), table, column)
    content["path"] = rel_path
    return ToolExecutionResult(ok="error" not in content, content=content)


def _make_execute_python(py_session: PersistentPythonSession) -> ToolHandler:
    """Return a handler that executes Python code inside a persistent session."""

    def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        if "code" not in action_input:
            return ToolExecutionResult(
                ok=False,
                content={
                    "success": False,
                    "error": "Missing required parameter 'code'.",
                    "output": "",
                    "stderr": "",
                },
            )
        code = str(action_input["code"])
        if py_session._context_root != task.context_dir.resolve():  # noqa: SLF001
            py_session._context_root = task.context_dir.resolve()  # noqa: SLF001
        content = py_session.execute(code, timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS)
        return ToolExecutionResult(ok=bool(content.get("success")), content=content)

    return _execute_python


def _calculate_math(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    expression = str(action_input["expression"])
    return ToolExecutionResult(ok=True, content=math_calculate(expression))


def _memory_grep_placeholder(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    del task, action_input
    return ToolExecutionResult(
        ok=False,
        content={
            "error": (
                "memory_grep is handled by the agent runtime because it searches "
                "per-task step history."
            )
        },
    )


def _make_extract_records(extractor: "RecordExtractor | None") -> ToolHandler:
    def _extract_records(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        if extractor is None:
            return ToolExecutionResult(
                ok=False, content={"ok": False, "error": "Record extractor is not configured."}
            )
        raw_path = action_input.get("path")
        if not raw_path:
            return ToolExecutionResult(
                ok=False, content={"ok": False, "error": "Missing required parameter 'path'."}
            )
        fields = action_input.get("fields")
        if isinstance(fields, str):
            fields = [f.strip() for f in fields.split(",") if f.strip()]
        if not isinstance(fields, list) or not fields:
            return ToolExecutionResult(
                ok=False,
                content={"ok": False, "error": "'fields' must be a non-empty list of field names."},
            )
        unit_hint = str(action_input.get("unit_hint", "record")).strip() or "record"
        doc_path = resolve_context_path(task, str(raw_path))
        content = extractor.extract(doc_path, fields=[str(f) for f in fields], unit_hint=unit_hint)
        return ToolExecutionResult(ok=bool(content.get("ok")), content=content)

    return _extract_records


def _make_inspect_video(inspector: VideoInspector | None) -> ToolHandler:
    def _inspect_video(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        if inspector is None:
            return ToolExecutionResult(
                ok=False,
                content={"ok": False, "error": "Video inspector is not configured."},
            )
        raw_path = action_input.get("path")
        if raw_path:
            video_path = resolve_context_path(task, str(raw_path))
        else:
            video_path = _find_first_video(task.context_dir)
            if video_path is None:
                return ToolExecutionResult(
                    ok=False,
                    content={"ok": False, "error": "No video file found in context."},
                )
        query = str(action_input.get("query", "")).strip()
        if not query:
            return ToolExecutionResult(
                ok=False,
                content={"ok": False, "error": "Missing required parameter 'query'."},
            )

        def _opt_float(key: str) -> float | None:
            val = action_input.get(key)
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        num_frames = int(action_input.get("num_frames", 12))
        content = inspector.inspect(
            video_path,
            query=query,
            start_time=_opt_float("start_time"),
            end_time=_opt_float("end_time"),
            num_frames=num_frames,
        )
        return ToolExecutionResult(ok=bool(content.get("ok")), content=content)

    return _inspect_video


_VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi"}


def _find_first_video(context_dir: Path) -> Path | None:
    if not context_dir.exists():
        return None
    for path in sorted(context_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _VIDEO_EXTS:
            return path
    return None


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,
        answer=answer,
    )


_SUBMIT_CSV_FILENAME = "answer.csv"


def _make_submit_csv(py_session: PersistentPythonSession) -> ToolHandler:
    """Terminating tool: submit the final answer from a CSV the agent wrote with
    pandas inside execute_python (``result.to_csv('answer.csv', index=False)``).
    This avoids forcing the model to inline hundreds of rows as JSON (which led to
    truncated/preview answers). The file lives in the execute_python sandbox cwd,
    so it is read back from the session work dir, not from the read-only /input."""

    def _submit_csv(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
        import csv as _csv  # noqa: PLC0415

        del action_input  # filename is fixed for robustness
        path = py_session.work_file_path(_SUBMIT_CSV_FILENAME)
        if path is None or not path.exists():
            return ToolExecutionResult(
                ok=False,
                content={
                    "status": "error",
                    "error": (
                        f"No '{_SUBMIT_CSV_FILENAME}' found in the Python working directory. "
                        f"First compute the FULL result in execute_python and write it with "
                        f"`result.to_csv('{_SUBMIT_CSV_FILENAME}', index=False)`, then call submit_csv."
                    ),
                },
            )
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(_csv.reader(handle))
        except OSError as exc:
            return ToolExecutionResult(
                ok=False,
                content={"status": "error", "error": f"Could not read {_SUBMIT_CSV_FILENAME}: {exc}"},
            )
        if not rows:
            return ToolExecutionResult(
                ok=False,
                content={
                    "status": "error",
                    "error": (
                        f"'{_SUBMIT_CSV_FILENAME}' is empty. Make sure your DataFrame is non-empty "
                        "and that the previous execute_python step succeeded before submitting."
                    ),
                },
            )
        columns = [str(c) for c in rows[0]]
        data_rows = [list(r) for r in rows[1:]]
        answer = AnswerTable(columns=columns, rows=data_rows)
        return ToolExecutionResult(
            ok=True,
            content={
                "status": "submitted",
                "source": _SUBMIT_CSV_FILENAME,
                "column_count": len(columns),
                "row_count": len(data_rows),
            },
            is_terminal=True,
            answer=answer,
        )

    return _submit_csv


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]
    _closer: Callable[[], None] | None = None

    def describe_for_prompt(self) -> str:
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(self, task: PublicTask, action: str, action_input: dict[str, Any]) -> ToolExecutionResult:
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input)

    def close(self) -> None:
        """Release resources held by this registry (e.g. the Python session subprocess)."""
        if self._closer is not None:
            try:
                self._closer()
            except Exception:  # noqa: BLE001
                pass


def create_default_tool_registry(
    context_root: "Path | None" = None,
    video_inspector: VideoInspector | None = None,
    record_extractor: RecordExtractor | None = None,
) -> ToolRegistry:
    """Create the default tool registry.

    A :class:`PersistentPythonSession` is created here and shared across all
    ``execute_python`` calls within this registry (i.e. within one task).
    Call :meth:`ToolRegistry.close` when the task finishes to terminate the
    underlying subprocess.

    ``video_inspector`` is a vision-to-text sub-agent used by ``inspect_video``;
    if None the tool reports that it is unavailable.
    """
    _root = (Path(context_root) if context_root else Path(".")).resolve()
    py_session = PersistentPythonSession(_root)

    specs: dict[str, ToolSpec] = {
        "answer": ToolSpec(
            name="answer",
            description=(
                "Submit the final answer table by INLINING rows. Best for small or "
                "aggregated results (a handful of rows). This is a terminating action. "
                "IMPORTANT: include ONLY the column(s) the question explicitly asks for - do "
                "not add extra identifier/context columns (e.g. year, id, name) unless the "
                "question asks for them. Use the original database/source column names. "
                "Return ALL matching rows, not a truncated preview. If the result has many "
                "rows (a long list of records), use `submit_csv` instead."
            ),
            input_schema={"columns": ["column_name"], "rows": [["value_1"]]},
        ),
        "submit_csv": ToolSpec(
            name="submit_csv",
            description=(
                "Submit the final answer from a CSV file you wrote in execute_python. "
                "Use this for results with MANY rows (a list of records) so you never have "
                "to inline rows or accidentally submit a truncated preview. Workflow: in "
                "execute_python compute the COMPLETE result as a DataFrame whose columns are "
                "exactly the requested answer column(s) with the original source names, then "
                "`result.to_csv('answer.csv', index=False)`; then call submit_csv (no args). "
                "It reads answer.csv and submits every row. This is a terminating action."
            ),
            input_schema={},
        ),
        "calculate_math": ToolSpec(
            name="calculate_math",
            description=(
                "Evaluate simple math expressions (add, subtract, multiply, divide, modulo, "
                "power). Use for arithmetic instead of writing Python."
            ),
            input_schema={"expression": "2 + 3 * 4"},
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description=(
                "Run a read-only SQL query against a sqlite/db file inside context. "
                "IMPORTANT: every context CSV file and every JSON file shaped like "
                "{table, records:[...]} is ALSO automatically available here as a table "
                "(table name = the JSON 'table' field, or the CSV filename without .csv), so "
                "you can query and JOIN CSV/JSON/sqlite data together in ONE SQL statement - "
                "no need for pandas merges. Use the table source index to pick the right table. "
                f"Default row limit is {DEFAULT_SQL_LIMIT}; raise `limit` if the result is "
                "truncated. For the final answer you must return the COMPLETE result set, "
                "not a truncated preview - check the `truncated` flag."
            ),
            input_schema={"path": "relative/path/to/file.sqlite", "sql": "SELECT ...", "limit": DEFAULT_SQL_LIMIT},
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute Python with the task context directory as the working directory. "
                "Returns captured stdout as `output`. The namespace PERSISTS across calls "
                "within a task (variables, imports, loaded DataFrames stay available). "
                f"Execution timeout is {EXECUTE_PYTHON_TIMEOUT_SECONDS}s. pandas/numpy/polars "
                "are available - use them to compute full aggregations for the final answer."
            ),
            input_schema={"code": "import pandas as pd\ndf = pd.read_csv('csv/data.csv')\nprint(df.head())"},
        ),
        "memory_grep": ToolSpec(
            name="memory_grep",
            description=(
                "Search your OWN past step history (original reasoning + tool observations) "
                "and get the full TEXT back. Use this to recover details that older steps "
                "lost to one-line summaries, e.g. the exact SQL you ran earlier, a precise "
                "error message, or the text of a past inspect_video reading. Returns text "
                "only - never raw video frames; to re-see frames, call inspect_video again "
                "on a narrow window."
            ),
            input_schema={
                "pattern": "regex/keyword to search past thoughts+observations (optional)",
                "step_index": "exact step number to fetch (optional)",
                "max_matches": 5,
            },
        ),
        "glob_context": ToolSpec(
            name="glob_context",
            description=(
                "Fast file pattern matching. Supports glob patterns like '**/*.csv', "
                "'data/*.json', '*.md'. Prefer over list_context when you know the file type."
            ),
            input_schema={"pattern": "**/*.csv", "path": "relative/subdir (optional)"},
        ),
        "grep_context": ToolSpec(
            name="grep_context",
            description=(
                "Regex content search across context files (works on large files without "
                "truncation). Filter file types with `include` (e.g. '*.md'). Use to locate "
                "specific keywords, corrected figures, or sections in documents."
            ),
            input_schema={
                "pattern": "regex, eg: corrected|adjusted",
                "path": "relative/path (optional)",
                "include": "*.md (optional glob filter)",
                "context_lines": 2,
                "max_matches": 50,
            },
        ),
        "profile_column": ToolSpec(
            name="profile_column",
            description=(
                "Profile ONE column of a table (CSV/JSON/db all queryable via the same "
                "`path` to the .sqlite) BEFORE you commit to it in an answer: returns row/"
                "distinct/null counts, sample values, numeric min/max/avg, and sibling "
                "columns - flagging same-theme look-alikes and DERIVED-stat columns "
                "(同类均值/排名/占比/avg/rank). Use this to confirm you picked the RAW value "
                "column the question wants, not a peer-average or ranking column with a "
                "similar name."
            ),
            input_schema={"path": "db/sub_db.sqlite", "table": "table_name", "column": "ColumnName"},
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite"},
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description=(
                "Read a PREVIEW of a CSV file inside context (default 50 rows). The returned "
                "`rows` are only a preview; `row_count` is the FULL number of rows in the file. "
                "NEVER submit these preview rows as the final answer - to return all rows, load "
                "the file in execute_python with pandas and use submit_csv. NOTE: a file with a "
                "matching column name but a surprisingly small row_count may be a DECOY; prefer "
                "the authoritative table named in the knowledge guide."
            ),
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 50},
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description=(
                "Read a text-like document (markdown/txt/html). Returns up to `limit` lines "
                "from line `offset` (1-indexed). Use the returned next_offset to page through "
                "large files. Use grep_context to locate content first."
            ),
            input_schema={"path": "relative/path/to/file.md", "offset": 1, "limit": 200},
        ),
        "read_pdf": ToolSpec(
            name="read_pdf",
            description=(
                "Extract text from a PDF file inside context as page-tagged markdown. "
                "Returns up to `max_chars` characters starting at `start_page` (1-indexed); "
                "use the returned `next_page` to page through long PDFs. Use this for any "
                ".pdf data source (e.g. doc/*.pdf) instead of read_doc."
            ),
            input_schema={"path": "relative/path/to/file.pdf", "start_page": 1, "max_chars": 12000},
        ),
        "extract_pdf_tables": ToolSpec(
            name="extract_pdf_tables",
            description=(
                "Recover STRUCTURED tables from a PDF (native row/column grids) instead of "
                "flattened text. Use this FIRST for any .pdf that holds a table/statement "
                "(balance sheets, rankings, multi-row records) - it preserves columns so you "
                "can read a specific field down the rows, unlike read_pdf which loses "
                "structure. Scans `max_pages` pages from `start_page`; returns each table as "
                "markdown with row/col counts and a `next_page` to continue. If it finds no "
                "tables (prose/scanned PDF), fall back to read_pdf."
            ),
            input_schema={"path": "relative/path/to/file.pdf", "start_page": 1, "max_pages": 8},
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 10000},
        ),
        "inspect_video": ToolSpec(
            name="inspect_video",
            description=(
                "Ask a vision sub-agent to WATCH the briefing video and answer a query in "
                "text. The briefing shows rules/thresholds/准入线/configurations as on-screen "
                "text and numbers. Pass a focused `query` (e.g. 'What is the free-float A-share "
                "threshold and the year qualifier?'). Optionally restrict to a time window with "
                "`start_time`/`end_time` (seconds) to read small digits precisely; omit them to "
                "scan the whole video first. Workflow: call once over the whole video to locate "
                "the rule, then call again on a narrow window to read exact numbers/operators. "
                "Returns TEXT only (no images enter your context)."
            ),
            input_schema={
                "query": "What threshold/rule does the video define?",
                "path": "relative/path/to/video.mp4 (optional, auto-detected)",
                "start_time": "0.0 (optional, seconds)",
                "end_time": "30.0 (optional, seconds)",
                "num_frames": 12,
            },
        ),
        "extract_records": ToolSpec(
            name="extract_records",
            description=(
                "Extract a LONG LIST of records from a NOISY PROSE document (PDF/markdown) "
                "whose data is NOT in a ruled table - the demo's prose PDFs hide real figures "
                "among distractor sentences and 'corrections'. A sub-agent reads the doc in "
                "chunks, extracts EVERY record per chunk with an evidence rule (only "
                "explicitly-stated values; uses the FINAL/corrected number on a correction "
                "trap; never substitutes a sibling field), then merges by unit id so nothing "
                "is dropped or duplicated. Pass `fields` (exact field names to pull per record) "
                "and a `unit_hint` (what one record is, e.g. 'one bank / one fund manager'). "
                "Returns columns+rows; review then submit via answer/submit_csv. Use this when "
                "read_pdf/extract_pdf_tables can't give you a clean per-row table."
            ),
            input_schema={
                "path": "doc/some_report.pdf",
                "fields": ["EntityName", "DepositsWithCentralBank"],
                "unit_hint": "one bank",
            },
        ),
    }

    handlers: dict[str, ToolHandler] = {
        "answer": _answer,
        "submit_csv": _make_submit_csv(py_session),
        "calculate_math": _calculate_math,
        "execute_context_sql": _execute_context_sql,
        "profile_column": _profile_column,
        "execute_python": _make_execute_python(py_session),
        "memory_grep": _memory_grep_placeholder,
        "glob_context": _glob_context,
        "grep_context": _grep_context,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_pdf": _read_pdf,
        "extract_pdf_tables": _extract_pdf_tables,
        "read_json": _read_json,
        "inspect_video": _make_inspect_video(video_inspector),
        "extract_records": _make_extract_records(record_extractor),
    }
    return ToolRegistry(specs=specs, handlers=handlers, _closer=py_session.close)
