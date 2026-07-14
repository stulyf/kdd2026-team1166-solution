from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


def _suggest_available_assets(task: PublicTask, relative_path: str) -> str:
    """Build a hint listing real available files when a path is not found, so the
    model can self-correct hallucinated names (e.g. asking for ``data.sqlite``
    when the real file is ``db/sub_db.sqlite``)."""
    context_root = task.context_dir.resolve()
    suffix = Path(relative_path).suffix.lower()
    try:
        all_files = [p.relative_to(context_root).as_posix() for p in context_root.rglob("*") if p.is_file()]
    except OSError:
        return ""
    same_ext = sorted(f for f in all_files if suffix and f.lower().endswith(suffix))
    pool = same_ext if same_ext else sorted(all_files)
    listed = pool[:25]
    hint = f" Available {suffix or 'files'}: {', '.join(listed)}" if listed else ""
    # CSV / SQLite are interchangeable sources here: a table missing as a CSV may
    # live inside a SQLite db (and vice versa). Nudge the model to check the db.
    if suffix in {".csv", ".sqlite", ".db", ".sqlite3"}:
        dbs = sorted(f for f in all_files if f.lower().endswith((".sqlite", ".db", ".sqlite3")))
        if dbs:
            hint += (
                f" NOTE: not every table is a CSV - a missing table may be a table "
                f"INSIDE a SQLite db ({', '.join(dbs[:5])}); inspect its schema with "
                f"inspect_sqlite_schema."
            )
    return hint


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    # LLM 有时会在路径前错误地加上 "context/" 前缀（因为 list_context 返回的 root 包含 context）
    # 自动去除这个冗余前缀以提高容错性
    cleaned_path = relative_path
    while cleaned_path.startswith("context/"):
        cleaned_path = cleaned_path[len("context/"):]

    candidate = (task.context_dir / cleaned_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(
            f"Missing context asset: {relative_path}.{_suggest_available_assets(task, relative_path)}"
        )
    return candidate


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, object]:
    entries: list[dict[str, object]] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }


def read_csv_preview(task: PublicTask, relative_path: str, *, max_rows: int = 50) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return {
            "path": relative_path,
            "columns": [],
            "rows": [],
            "row_count": 0,
        }

    header = rows[0]
    data_rows = rows[1:]
    return {
        "path": relative_path,
        "columns": header,
        "rows": data_rows[:max_rows],
        "row_count": len(data_rows),
        "truncated": len(data_rows) > max_rows,
    }


def read_json_preview(task: PublicTask, relative_path: str, *, max_chars: int = 10000) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    payload = json.loads(path.read_text())
    preview = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "path": relative_path,
        "preview": preview[:max_chars],
        "truncated": len(preview) > max_chars,
    }


_READ_DEFAULT_LIMIT = 200   # lines per call
_READ_MAX_BYTES = 50 * 1024  # 50 KB hard cap
_READ_MAX_LINE_LEN = 2000    # chars per line before truncation


def read_doc_preview(
    task: PublicTask,
    relative_path: str,
    *,
    offset: int = 1,
    limit: int = _READ_DEFAULT_LIMIT,
) -> dict[str, object]:
    """Read a text document with line-based pagination."""
    path = resolve_context_path(task, relative_path)
    all_lines = path.read_text(errors="replace").splitlines()
    total_lines = len(all_lines)

    start = max(0, offset - 1)          # convert to 0-indexed
    raw: list[str] = []
    bytes_used = 0
    cut = False

    for line in all_lines[start : start + limit]:
        if len(line) > _READ_MAX_LINE_LEN:
            line = line[:_READ_MAX_LINE_LEN] + f"... (line truncated to {_READ_MAX_LINE_LEN} chars)"
        size = len(line.encode("utf-8")) + 1  # +1 for newline
        if bytes_used + size > _READ_MAX_BYTES:
            cut = True
            break
        raw.append(line)
        bytes_used += size

    last_line = start + len(raw)        # 0-indexed exclusive
    more = last_line < total_lines
    truncated = cut or (more and len(raw) >= limit)
    next_offset = last_line + 1 if truncated else None  # 1-indexed

    numbered = "\n".join(f"{start + i + 1}: {line}" for i, line in enumerate(raw))

    if cut:
        suffix = f"\n\n(Output capped at 50 KB. Lines {offset}-{last_line}. Use offset={next_offset} to continue.)"
    elif truncated:
        suffix = f"\n\n(Showing lines {offset}-{last_line} of {total_lines}. Use offset={next_offset} to continue.)"
    else:
        suffix = f"\n\n(End of file - {total_lines} lines total.)"

    return {
        "path": relative_path,
        "preview": numbered + suffix,
        "offset": offset,
        "lines_returned": len(raw),
        "total_lines": total_lines,
        "truncated": truncated,
        "next_offset": next_offset,
    }


_PDF_DEFAULT_MAX_CHARS = 12000


def read_pdf_preview(
    task: PublicTask,
    relative_path: str,
    *,
    start_page: int = 1,
    max_chars: int = _PDF_DEFAULT_MAX_CHARS,
) -> dict[str, object]:
    """Extract text from a PDF (text-based, not OCR) as page-tagged markdown.

    Supports page pagination via ``start_page`` and a per-call ``max_chars`` cap
    so multi-page financial PDFs do not blow up the context. Returns ``next_page``
    when more pages remain so the model can continue reading."""
    try:
        import fitz  # PyMuPDF  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dependency guard
        return {
            "path": relative_path,
            "error": (
                "PyMuPDF (fitz) is not installed, so PDF text cannot be extracted. "
                "Try read_doc if a parsed text version exists, or use execute_python."
            ),
            "detail": str(exc),
        }

    path = resolve_context_path(task, relative_path)
    doc = fitz.open(path)
    try:
        total_pages = doc.page_count
        start_idx = max(0, start_page - 1)
        parts: list[str] = []
        chars_used = 0
        last_page = start_idx  # 0-indexed exclusive
        truncated = False
        for page_idx in range(start_idx, total_pages):
            text = doc[page_idx].get_text()
            block = f"## Page {page_idx + 1}\n\n{text}".strip()
            if parts and chars_used + len(block) > max_chars:
                truncated = True
                break
            parts.append(block)
            chars_used += len(block)
            last_page = page_idx + 1
            if chars_used >= max_chars:
                truncated = len(text) > 0 and last_page < total_pages
                break
    finally:
        doc.close()

    more = last_page < total_pages
    next_page = last_page + 1 if more else None
    preview = "\n\n".join(parts)
    if more:
        preview += (
            f"\n\n(Showing pages {start_page}-{last_page} of {total_pages}. "
            f"Use start_page={next_page} to continue.)"
        )
    else:
        preview += f"\n\n(End of PDF - {total_pages} pages total.)"

    return {
        "path": relative_path,
        "preview": preview,
        "start_page": start_page,
        "pages_returned": last_page - start_idx,
        "total_pages": total_pages,
        "truncated": truncated or more,
        "next_page": next_page,
    }


def extract_pdf_tables(
    task: PublicTask,
    relative_path: str,
    *,
    start_page: int = 1,
    max_pages: int = 8,
    max_chars: int = _PDF_DEFAULT_MAX_CHARS,
) -> dict[str, object]:
    """Recover STRUCTURED tables from a PDF using PyMuPDF's native ``find_tables``.

    Unlike ``read_pdf`` (which flattens a page to running text and loses row/column
    structure), this detects ruled/whitespace tables and returns each as a markdown
    grid plus row/column counts, so financial statements (balance sheets, manager
    rankings) come back as real tables the model can read column-wise.

    PURE PyMuPDF - no OCR, no ML models, no downloads. Best on native (text-based)
    PDFs; on a scanned PDF it finds nothing and you should fall back to read_pdf."""
    try:
        import fitz  # PyMuPDF  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dependency guard
        return {
            "path": relative_path,
            "error": (
                "PyMuPDF (fitz) is not installed, so PDF tables cannot be extracted. "
                "Try read_pdf for raw text, or execute_python."
            ),
            "detail": str(exc),
        }

    path = resolve_context_path(task, relative_path)
    doc = fitz.open(path)
    tables_out: list[dict[str, object]] = []
    chars_used = 0
    truncated = False
    last_page = start_page - 1
    try:
        total_pages = doc.page_count
        start_idx = max(0, start_page - 1)
        end_idx = min(total_pages, start_idx + max_pages)
        for page_idx in range(start_idx, end_idx):
            last_page = page_idx + 1
            page = doc[page_idx]
            try:
                finder = page.find_tables()
            except Exception:  # noqa: BLE001 - find_tables can raise on odd pages
                continue
            for ti, table in enumerate(getattr(finder, "tables", []) or []):
                try:
                    grid = table.extract()
                except Exception:  # noqa: BLE001
                    continue
                if not grid:
                    continue
                md = _grid_to_markdown(grid)
                if chars_used + len(md) > max_chars and tables_out:
                    truncated = True
                    break
                tables_out.append(
                    {
                        "page": page_idx + 1,
                        "table_index": ti,
                        "n_rows": len(grid),
                        "n_cols": max((len(r) for r in grid), default=0),
                        "markdown": md,
                    }
                )
                chars_used += len(md)
            if truncated:
                break
    finally:
        doc.close()

    result: dict[str, object] = {
        "path": relative_path,
        "tables_found": len(tables_out),
        "tables": tables_out,
        "pages_scanned": f"{start_page}-{last_page}",
        "truncated": truncated,
        "next_page": last_page + 1 if truncated else None,
    }
    if not tables_out:
        result["note"] = (
            "No ruled/whitespace tables detected on the scanned pages. This may be a "
            "prose-style PDF or scanned image - use read_pdf for the raw text instead."
        )
    return result


def _grid_to_markdown(grid: list[list[object]]) -> str:
    """Render a table grid (list of rows) as a compact markdown table."""
    def _cell(v: object) -> str:
        return ("" if v is None else str(v)).replace("\n", " ").replace("|", "\\|").strip()

    rows = [[_cell(c) for c in row] for row in grid]
    width = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (width - len(r)) for r in rows]
    if not rows:
        return ""
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# glob_context — fast file pattern matching across context files
# ---------------------------------------------------------------------------

def glob_context(task: PublicTask, pattern: str, *, path: str | None = None) -> dict[str, object]:
    """Find files by glob pattern inside context, sorted by mtime (newest first)."""
    if path:
        search_root = resolve_context_path(task, path)
    else:
        search_root = task.context_dir.resolve()

    matches = sorted(search_root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    rel_paths = [str(p.relative_to(task.context_dir.resolve())) for p in matches if p.is_file()]

    return {
        "pattern": pattern,
        "search_root": str(search_root.relative_to(task.context_dir.resolve())) if path else ".",
        "match_count": len(rel_paths),
        "matches": rel_paths,
    }


# ---------------------------------------------------------------------------
# grep_context — fast regex search across context files
# ---------------------------------------------------------------------------

_RG_AVAILABLE: bool | None = None  # lazily detected


def _rg_available() -> bool:
    global _RG_AVAILABLE
    if _RG_AVAILABLE is None:
        _RG_AVAILABLE = shutil.which("rg") is not None
    return _RG_AVAILABLE


def _grep_with_rg(
    search_root: Path,
    pattern: str,
    *,
    include: str | None,
    context_lines: int,
    max_matches: int,
) -> list[dict[str, object]]:
    args = [
        "rg",
        "--no-config",
        "--json",
        "--no-messages",
        "--hidden",
        "--glob=!.git/*",
        f"--context={context_lines}",
        f"--max-count={max_matches}",
    ]
    if include:
        args.append(f"--glob={include}")
    args += ["--", pattern, str(search_root)]

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return []

    matches: list[dict[str, object]] = []
    context_before: list[str] = []

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type")
        if msg_type == "context":
            data = obj.get("data", {})
            context_before.append(data.get("lines", {}).get("text", "").rstrip("\n"))
        elif msg_type == "match":
            data = obj.get("data", {})
            rel = str(Path(data.get("path", {}).get("text", "")).relative_to(search_root))
            matches.append(
                {
                    "file": rel,
                    "line_number": data.get("line_number"),
                    "match": data.get("lines", {}).get("text", "").rstrip("\n"),
                    "context_before": list(context_before),
                }
            )
            context_before = []
            if len(matches) >= max_matches:
                break
        else:
            context_before = []

    return matches


def _grep_with_python(
    search_root: Path,
    pattern: str,
    *,
    include: str | None,
    context_lines: int,
    max_matches: int,
) -> list[dict[str, object]]:
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {exc}") from exc

    if include:
        candidates = list(search_root.rglob(include))
    else:
        candidates = [p for p in search_root.rglob("*") if p.is_file()]

    matches: list[dict[str, object]] = []
    for filepath in sorted(candidates):
        if not filepath.is_file():
            continue
        try:
            file_lines = filepath.read_text(errors="replace").splitlines()
        except OSError:
            continue
        rel = str(filepath.relative_to(search_root))
        for i, text in enumerate(file_lines):
            if compiled.search(text):
                start = max(0, i - context_lines)
                matches.append(
                    {
                        "file": rel,
                        "line_number": i + 1,
                        "match": text,
                        "context_before": file_lines[start:i],
                    }
                )
                if len(matches) >= max_matches:
                    return matches
    return matches


def grep_context(
    task: PublicTask,
    pattern: str,
    *,
    path: str | None = None,
    include: str | None = None,
    context_lines: int = 2,
    max_matches: int = 50,
) -> dict[str, object]:
    """Search for a regex pattern across context files (ripgrep when available)."""
    if path:
        search_root = resolve_context_path(task, path)
    else:
        search_root = task.context_dir.resolve()

    if _rg_available():
        matches = _grep_with_rg(
            search_root, pattern, include=include,
            context_lines=context_lines, max_matches=max_matches,
        )
        backend = "ripgrep"
    else:
        matches = _grep_with_python(
            search_root, pattern, include=include,
            context_lines=context_lines, max_matches=max_matches,
        )
        backend = "python-re"

    return {
        "pattern": pattern,
        "search_root": str(search_root.relative_to(task.context_dir.resolve()) if path else "."),
        "match_count": len(matches),
        "truncated": len(matches) >= max_matches,
        "backend": backend,
        "matches": matches,
    }
