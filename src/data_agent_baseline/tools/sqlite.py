from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def inspect_sqlite_schema(path: Path) -> dict[str, object]:
    with _connect_read_only(path) as conn:
        rows = conn.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        tables: list[dict[str, object]] = []
        for name, create_sql in rows:
            tables.append(
                {
                    "name": name,
                    "create_sql": create_sql,
                }
            )
    return {
        "path": str(path),
        "tables": tables,
    }


def execute_read_only_sql(path: Path, sql: str, *, limit: int = 200) -> dict[str, object]:
    normalized_sql = sql.lstrip().lower()
    if not normalized_sql.startswith(("select", "with", "pragma")):
        raise ValueError("Only read-only SQL statements are allowed.")

    with _connect_read_only(path) as conn:
        cursor = conn.execute(sql)
        column_names = [item[0] for item in cursor.description or []]
        rows = cursor.fetchmany(limit + 1)

    truncated = len(rows) > limit
    limited_rows = rows[:limit]
    return {
        "path": str(path),
        "columns": column_names,
        "rows": [list(row) for row in limited_rows],
        "row_count": len(limited_rows),
        "truncated": truncated,
    }


def _tokens(name: str) -> set[str]:
    import re as _re

    return {t for t in _re.split(r"[^0-9a-zA-Z\u4e00-\u9fff]+", name.lower()) if t}


# Column-name tokens that mark a PRE-DERIVED statistic rather than a raw value.
# Surfaced so the agent (and verifier) can tell "总管理规模" from "总管理规模同类均值".
_DERIVED_HINT_TOKENS = {
    "均值", "平均", "中位数", "同类", "排名", "排序", "占比", "比例",
    "avg", "average", "mean", "median", "rank", "ranking", "pct", "percent", "ratio",
}


def profile_column(path: Path, table: str, column: str) -> dict[str, object]:
    """Profile one column so the agent can confirm it picked the RIGHT column
    before answering: row/distinct/null counts, sample values, numeric range, and
    sibling columns (flagging same-theme look-alikes and derived-statistic columns
    like '同类均值'/'排名' that are common wrong picks for a raw-value aggregate)."""
    with _connect_read_only(path) as conn:
        cols_info = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        if not cols_info:
            return {"table": table, "error": f"Table '{table}' not found or has no columns."}
        col_names = [c[1] for c in cols_info]
        col_types = {c[1]: (c[2] or "").upper() for c in cols_info}
        if column not in col_names:
            return {
                "table": table,
                "column": column,
                "error": f"Column '{column}' not in table. Available: {col_names}",
            }

        total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        non_null = conn.execute(
            f'SELECT COUNT("{column}") FROM "{table}"'
        ).fetchone()[0]
        distinct = conn.execute(
            f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"'
        ).fetchone()[0]
        samples = [
            r[0]
            for r in conn.execute(
                f'SELECT DISTINCT "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL LIMIT 8'
            ).fetchall()
        ]
        numeric_range = None
        if col_types.get(column) in {"INTEGER", "REAL", "NUMERIC"}:
            row = conn.execute(
                f'SELECT MIN("{column}"), MAX("{column}"), AVG("{column}") FROM "{table}"'
            ).fetchone()
            if row is not None:
                numeric_range = {"min": row[0], "max": row[1], "avg": row[2]}

    target_tokens = _tokens(column)
    siblings: list[dict[str, object]] = []
    for name in col_names:
        if name == column:
            continue
        shared = target_tokens & _tokens(name)
        is_derived = bool(_tokens(name) & {t.lower() for t in _DERIVED_HINT_TOKENS}) or any(
            h in name for h in _DERIVED_HINT_TOKENS
        )
        if shared or is_derived:
            siblings.append(
                {
                    "column": name,
                    "type": col_types.get(name, ""),
                    "same_theme": bool(shared),
                    "looks_derived": is_derived,
                }
            )

    target_is_derived = any(h in column for h in _DERIVED_HINT_TOKENS) or bool(
        _tokens(column) & {t.lower() for t in _DERIVED_HINT_TOKENS}
    )
    return {
        "table": table,
        "column": column,
        "column_type": col_types.get(column, ""),
        "row_count": total,
        "non_null": non_null,
        "null_rate": round(1 - (non_null / total), 4) if total else None,
        "distinct": distinct,
        "sample_values": samples,
        "numeric_range": numeric_range,
        "looks_derived": target_is_derived,
        "related_columns": siblings,
        "hint": (
            "This column name looks like a DERIVED statistic (mean/rank/ratio). If the "
            "question asks for a raw value or its max/sum/total, prefer the raw base "
            "column from related_columns instead."
            if target_is_derived
            else "Compare numeric_range/sample_values against the question to confirm this "
            "is the intended column (raw value vs a derived/ranked sibling)."
        ),
    }


# ---------------------------------------------------------------------------
# In-place SQLite augmentation: load context CSV / JSON files as tables so that
# a single SQL query can reach every data source. The augmented copy is written
# to a writable /tmp location (NEVER the read-only /input), keyed by source DB
# so it is built once per task and reused.
# ---------------------------------------------------------------------------

_AUGMENTED_DB_CACHE: dict[str, Path] = {}


def _infer_type_from_values(values: list[object]) -> str:
    types_found: set[str] = set()
    for value in values:
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            types_found.add("INTEGER")
        elif isinstance(value, int):
            types_found.add("INTEGER")
        elif isinstance(value, float):
            types_found.add("REAL")
        else:
            types_found.add("TEXT")
    if not types_found or "TEXT" in types_found:
        return "TEXT"
    if "REAL" in types_found:
        return "REAL"
    return "INTEGER"


def _coerce(value: object, sql_type: str) -> object:
    if value is None or value == "":
        return None
    if sql_type == "INTEGER":
        try:
            return int(float(value))  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return None
    if sql_type == "REAL":
        try:
            return float(value)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return None
    return str(value)


def _load_csv_to_table(conn: sqlite3.Connection, csv_path: Path, table_name: str) -> int:
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return 0
    header = rows[0]
    data_rows = rows[1:]

    columns: list[tuple[str, str]] = []
    for col_idx, col_name in enumerate(header):
        col_values = [row[col_idx] if col_idx < len(row) else "" for row in data_rows]
        columns.append((str(col_name), _infer_type_from_values(col_values)))

    col_defs = ", ".join(f'"{name}" {typ}' for name, typ in columns)
    conn.execute(f'CREATE TABLE "{table_name}" ({col_defs})')
    placeholders = ", ".join("?" for _ in columns)
    for row in data_rows:
        typed_row = [
            _coerce(row[idx] if idx < len(row) else None, typ)
            for idx, (_, typ) in enumerate(columns)
        ]
        conn.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders})', typed_row)
    conn.commit()
    return len(data_rows)


def _load_json_to_table(conn: sqlite3.Connection, json_path: Path) -> str | None:
    """Load a JSON file shaped like ``{"table": name, "records": [...]}`` into a
    table. Returns the created table name, or None if the file is not that shape."""
    payload = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(payload, dict):
        return None
    table_name = payload.get("table")
    records = payload.get("records")
    if not table_name or not isinstance(records, list) or not records:
        return None

    all_keys: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            return None
        for key in record:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    columns = [
        (key, _infer_type_from_values([rec.get(key) for rec in records]))
        for key in all_keys
    ]
    col_defs = ", ".join(f'"{name}" {typ}' for name, typ in columns)
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})')
    placeholders = ", ".join("?" for _ in columns)
    for record in records:
        typed_row = [_coerce(record.get(name), typ) for name, typ in columns]
        conn.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders})', typed_row)
    conn.commit()
    return str(table_name)


def augment_sqlite_db(context_dir: Path, db_abs_path: Path) -> Path:
    """Return a writable copy of ``db_abs_path`` with every context CSV / JSON
    file added as a table (when a same-named table is not already present).

    The copy lives under a /tmp directory so the original read-only /input is
    never modified. Results are cached per source DB for the duration of the
    process.
    """
    cache_key = str(db_abs_path.resolve())
    cached = _AUGMENTED_DB_CACHE.get(cache_key)
    if cached is not None and cached.exists():
        return cached

    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:12]
    target_dir = Path(tempfile.gettempdir()) / "dabench_augmented" / digest
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    augmented_path = target_dir / db_abs_path.name
    augmented_path.write_bytes(db_abs_path.read_bytes())

    conn = sqlite3.connect(str(augmented_path))
    try:
        existing = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        # Skip CSV/JSON sitting directly in the context root: those loose files are
        # almost always pre-computed DECOYS (e.g. trading_volume_601908.csv,
        # *_result.csv) whose values are corrupted/mis-columned. Exposing them as
        # SQL tables makes them the path of least resistance ("SELECT * FROM
        # <decoy>"). Only authoritative tables under csv/ json/ db/ doc/ are loaded.
        for json_file in sorted(context_dir.rglob("*.json")):
            if json_file.parent == context_dir:
                continue
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                continue
            if isinstance(payload, dict) and payload.get("table") and payload["table"] not in existing:
                name = _load_json_to_table(conn, json_file)
                if name:
                    existing.add(name)
        for csv_file in sorted(context_dir.rglob("*.csv")):
            if csv_file.parent == context_dir:
                continue
            table_name = csv_file.stem
            if table_name in existing:
                continue
            try:
                if _load_csv_to_table(conn, csv_file, table_name) > 0:
                    existing.add(table_name)
            except (OSError, ValueError, sqlite3.Error):
                continue
    finally:
        conn.close()

    _AUGMENTED_DB_CACHE[cache_key] = augmented_path
    return augmented_path
