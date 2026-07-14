"""Schema-guided record extraction from noisy prose documents.

The KDD demo ships many tasks whose gold answer is a long list of records (e.g.
50 rows) buried in an adversarial PROSE PDF/markdown - no ruled table, lots of
distractor sentences and deliberate "corrections". PyMuPDF's table finder gets
nothing on these (0/155 demo PDFs have a detectable table), so structure has to
be recovered by reading.

`RecordExtractor` implements a "extract-per-chunk then merge/dedup" pass
(inspired by TST's text-to-table count-then-extract): the document is split into
paragraph chunks, each chunk is asked to return EVERY record it contains with an
evidence rule (only values explicitly stated), then records are merged by a
stable unit key so nothing is dropped or double-counted.

It uses the SAME declared answer model via a plain OpenAI client - it introduces
NO extra model (no OCR/vision/layout weights), only more calls to the configured
LLM, exactly like the planner / answer-verifier sub-agents.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_EXTRACT_PROMPT = """
You extract STRUCTURED records from a chunk of a NOISY PROSE document. The prose
hides real figures among distractor sentences (drills, landscaping, courier
changes, 绿化/消防/etc.) and sometimes states a wrong value first then "corrects"
it.

You are given a RECORD UNIT description and a list of REQUESTED FIELDS. Extract
EVERY record of that unit present in THIS chunk - do not skip any, do not invent
records not in the chunk.

Rules:
- Output ONLY the requested fields for each record found in this chunk.
- Numbers may be comma-grouped (1,817,970) or Chinese-formatted -> output a plain
  number (strip commas/units).
- CORRECTION TRAP: if both a preliminary/estimated/approximate AND a final/
  audited/reconciled/corrected value appear for the same field, use the FINAL/
  corrected one ONLY.
- Do NOT substitute a sibling or derived field. Match the EXACT requested field
  name (e.g. DepositsWithCentralBank is NOT ReserveAssets or CashInVault; a raw
  value is NOT its peer-average or rank).
- EVIDENCE RULE: only output a value that is explicitly stated in THIS chunk for
  that record; otherwise use null.
- "unit" is a stable identifier for the record (its id/code/name), so records can
  be merged across chunks without duplication.

Output exactly ONE JSON object, no markdown:
{"records": [{"unit": "<id>", "<field1>": <number|string|null>, ...}, ...]}
""".strip()


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _chunks(paras: list[str], size: int) -> list[str]:
    return ["\n\n".join(paras[i : i + size]) for i in range(0, len(paras), size)]


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz  # noqa: PLC0415
        except ImportError:
            return ""
        doc = fitz.open(path)
        try:
            return "\n\n".join(doc[i].get_text() for i in range(doc.page_count))
        finally:
            doc.close()
    return path.read_text(encoding="utf-8", errors="replace")


class RecordExtractor:
    """Read a prose doc and extract a list of records for the requested schema."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        chunk_paragraphs: int = 28,
        max_chunks: int = 12,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chunk_paragraphs = chunk_paragraphs
        self.max_chunks = max_chunks
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # noqa: PLC0415

            self._client = OpenAI(
                api_key=self.api_key, base_url=self.api_base, timeout=600.0, max_retries=2
            )
        return self._client

    def _extract_chunk(self, chunk: str, fields: list[str], unit_hint: str) -> list[dict[str, Any]]:
        user = (
            f"RECORD UNIT: {unit_hint}\n"
            f"REQUESTED FIELDS: {', '.join(fields)}\n\n"
            f"Document chunk:\n{chunk}\n\nExtract the JSON now."
        )
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _EXTRACT_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = resp.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            return []
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            return []
        records = payload.get("records") if isinstance(payload, dict) else None
        return records if isinstance(records, list) else []

    def extract(
        self,
        doc_path: Path,
        *,
        fields: list[str],
        unit_hint: str = "record",
    ) -> dict[str, Any]:
        text = _read_text(doc_path)
        if not text.strip():
            return {"ok": False, "error": f"Could not read text from {doc_path.name}."}
        chunks = _chunks(_paragraphs(text), self.chunk_paragraphs)
        if len(chunks) > self.max_chunks:
            return {
                "ok": False,
                "error": (
                    f"Document has {len(chunks)} chunks (> {self.max_chunks}); too large for "
                    "one extraction pass. Narrow the doc or raise max_chunks."
                ),
            }

        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        unitless = 0
        for ci, chunk in enumerate(chunks):
            for rec in self._extract_chunk(chunk, fields, unit_hint):
                if not isinstance(rec, dict):
                    continue
                # Skip records that carry no requested field value at all.
                if all(rec.get(f) in (None, "") for f in fields):
                    continue
                unit = str(rec.get("unit", "")).strip()
                if not unit:
                    # Never silently drop a record that has data but no unit id:
                    # give it a synthetic key so it still surfaces (merge/dedup
                    # across chunks just won't apply to it).
                    unitless += 1
                    unit = f"__row_{ci}_{unitless}"
                if unit not in merged:
                    merged[unit] = {}
                    order.append(unit)
                slot = merged[unit]
                for f in fields:
                    val = rec.get(f)
                    if val is not None and val != "" and slot.get(f) is None:
                        slot[f] = val

        rows = [[merged[u].get(f) for f in fields] for u in order]
        complete = [r for r in rows if all(c is not None for c in r)]
        display_units = ["" if u.startswith("__row_") else u for u in order]
        return {
            "ok": True,
            "doc": doc_path.name,
            "unit_hint": unit_hint,
            "fields": fields,
            "n_units": len(order),
            "n_complete_rows": len(complete),
            "units": display_units,
            "columns": ["unit", *fields],
            "rows": [[display_units[i], *rows[i]] for i in range(len(order))],
            "note": (
                "Records merged across chunks by unit id. Review the rows, then submit the "
                "field column(s) the question asks for via answer/submit_csv. Rows with a "
                "null mean that field was not found for that unit."
            ),
        }
