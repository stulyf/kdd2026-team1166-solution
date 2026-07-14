from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

# Files that ``execute_python`` might OPEN FOR WRITING (e.g. ``pd.to_sql`` into an
# existing sqlite). These are copied into the scratch mirror so writes land on the
# disposable copy instead of polluting the read-only ``/input`` source. Everything
# else is symlinked (zero-copy) since it is only ever read.
_WRITABLE_SUFFIXES = {".sqlite", ".db", ".sqlite3", ".duckdb"}


def _build_scratch_mirror(context_root: Path) -> Path:
    """Build a writable mirror of ``context_root`` under a temp dir.

    The mirror reproduces the directory structure so relative paths used by the
    agent (e.g. ``csv/foo.csv``, ``db/sub_db.sqlite``) keep working, but any write
    performed by ``execute_python`` (``to_sql``, new ``.db`` files, output files)
    lands inside the disposable mirror rather than the shared ``/input`` source.

    - sqlite/db files are COPIED (so in-place writes hit the copy).
    - all other files are SYMLINKED (read-only, zero-copy even for large CSVs/videos).
    """
    mirror = Path(tempfile.mkdtemp(prefix="dabench_pyexec_"))
    if not context_root.exists():
        return mirror
    for src in context_root.rglob("*"):
        rel = src.relative_to(context_root)
        dest = mirror / rel
        if src.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if src.suffix.lower() in _WRITABLE_SUFFIXES:
                shutil.copy2(src, dest)
            else:
                dest.symlink_to(src.resolve())
        except OSError:
            # Symlink unsupported / copy failed -> best-effort fallback to copy.
            try:
                shutil.copy2(src, dest)
            except OSError:
                pass
    return mirror

# ---------------------------------------------------------------------------
# Worker script embedded as a string.
# Launched as `python -c <script>` in a subprocess per task.
# Protocol (newline-delimited JSON over stdin/stdout):
#   stdin  -> {"code": "...", "cwd": "..."} | {"exit": true}
#   stdout <- {"success": bool, "output": "...", "stderr": "...",
#              "error": "...|null", "traceback": "...|null"}
# ---------------------------------------------------------------------------
_WORKER_SCRIPT = r"""
import sys, json, os, io, traceback
from pathlib import Path

if hasattr(sys.stdin, 'reconfigure'):
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

_NS: dict = {"__builtins__": __builtins__, "__name__": "__main__", "Path": Path}

for _raw in sys.stdin:
    _raw = _raw.strip()
    if not _raw:
        continue
    try:
        _msg = json.loads(_raw)
    except Exception:
        continue

    if _msg.get("exit"):
        break

    if "cwd" in _msg:
        try:
            os.chdir(_msg["cwd"])
        except Exception:
            pass

    _code = _msg.get("code", "")
    _out_buf = io.StringIO()
    _err_buf = io.StringIO()
    _real_stdout = sys.stdout
    _real_stderr = sys.stderr
    sys.stdout = _out_buf
    sys.stderr = _err_buf
    try:
        exec(_code, _NS, _NS)  # noqa: S102
        _result = {
            "success": True,
            "output": _out_buf.getvalue(),
            "stderr": _err_buf.getvalue(),
            "error": None,
            "traceback": None,
        }
    except BaseException as _exc:  # noqa: BLE001
        _result = {
            "success": False,
            "output": _out_buf.getvalue(),
            "stderr": _err_buf.getvalue(),
            "error": str(_exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        sys.stdout = _real_stdout
        sys.stderr = _real_stderr

    _real_stdout.write(json.dumps(_result) + "\n")
    _real_stdout.flush()
"""


class PersistentPythonSession:
    """A per-task Python subprocess whose namespace survives across calls.

    - Created once at the start of a task (lazy, on first execute_python call).
    - All execute() calls within the same task share the same namespace.
    - Closed (subprocess terminated) when close() is called at task end.
    - After a timeout the subprocess is automatically killed and restarted.
    """

    def __init__(self, context_root: Path) -> None:
        self._context_root = context_root.resolve()
        self._active_root: Path | None = None  # root the current mirror was built for
        self._work_dir: Path | None = None  # writable scratch mirror (worker cwd)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._initialized = False  # cwd sent to worker?
        self._start()

    def _ensure_work_dir(self) -> None:
        """Build (or rebuild on task change) the writable scratch mirror."""
        if self._work_dir is not None and self._active_root == self._context_root:
            return
        self._cleanup_work_dir()
        self._work_dir = _build_scratch_mirror(self._context_root)
        self._active_root = self._context_root
        self._initialized = False  # force the new cwd to be sent to the worker

    def _cleanup_work_dir(self) -> None:
        if self._work_dir is not None:
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._work_dir = None
            self._active_root = None

    def work_file_path(self, name: str) -> Path | None:
        """Return the absolute path of ``name`` inside the writable scratch mirror
        (the cwd that ``execute_python`` runs in). Used by ``submit_csv`` to read
        an answer file the agent wrote with pandas ``to_csv``."""
        with self._lock:
            self._ensure_work_dir()
            if self._work_dir is None:
                return None
            return self._work_dir / name

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._initialized = False

    def _restart(self) -> None:
        try:
            if self._proc is not None:
                self._proc.kill()
                self._proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            pass
        self._start()

    def _send_code(self, code: str) -> None:
        assert self._proc is not None
        msg: dict[str, Any] = {"code": code}
        if not self._initialized:
            target = self._work_dir or self._context_root
            msg["cwd"] = str(target)
            self._initialized = True
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def execute(self, code: str, *, timeout_seconds: int = 30) -> dict[str, Any]:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()

            self._ensure_work_dir()

            try:
                self._send_code(code)
            except OSError:
                self._restart()
                return {
                    "success": False,
                    "output": "",
                    "stderr": "",
                    "error": "Python session restarted (broken pipe).",
                    "traceback": None,
                }

            result_holder: list[str] = []

            def _read() -> None:
                try:
                    assert self._proc is not None
                    line = self._proc.stdout.readline()
                    if line:
                        result_holder.append(line)
                except Exception:  # noqa: BLE001
                    pass

            reader = threading.Thread(target=_read, daemon=True)
            reader.start()
            reader.join(timeout_seconds)

            if reader.is_alive() or not result_holder:
                self._restart()
                return {
                    "success": False,
                    "output": "",
                    "stderr": "",
                    "error": (
                        f"Python execution timed out after {timeout_seconds}s. "
                        "Session restarted - prior variables are no longer available."
                    ),
                    "traceback": None,
                }

            try:
                return json.loads(result_holder[0])
            except Exception:  # noqa: BLE001
                return {
                    "success": False,
                    "output": result_holder[0] if result_holder else "",
                    "stderr": "",
                    "error": "Failed to parse worker response.",
                    "traceback": None,
                }

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.write('{"exit": true}\n')
                self._proc.stdin.flush()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                pass
            self._proc = None
        self._cleanup_work_dir()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
