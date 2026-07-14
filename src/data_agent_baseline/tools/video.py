from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any

_MAX_FRAMES = 24
_JPEG_QUALITY = 90
_MAX_EDGE = 1280  # cap longest edge to control payload size

_LOW_CONFIDENCE_THRESHOLD = 0.6  # below this we auto re-read once with denser/narrower sampling


def _probe_duration(container: Any) -> float:
    """Return video duration in seconds (best-effort)."""
    if container.duration is not None:
        return float(container.duration) / 1_000_000.0
    stream = container.streams.video[0]
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    return 0.0


def _encode_frame_jpeg(frame: Any) -> str:
    """Convert a PyAV frame to a base64-encoded JPEG (downscaled if large)."""
    image = frame.to_image()  # PIL.Image
    w, h = image.size
    longest = max(w, h)
    if longest > _MAX_EDGE:
        scale = _MAX_EDGE / longest
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def extract_frames(
    video_path: Path,
    *,
    num_frames: int = 12,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[list[str], list[float], float]:
    """Extract evenly-spaced frames within [start_time, end_time].

    Returns (base64_jpegs, timestamps_sec, duration_sec).
    """
    import av  # noqa: PLC0415

    num_frames = max(1, min(int(num_frames), _MAX_FRAMES))
    images_b64: list[str] = []
    timestamps: list[float] = []

    with av.open(str(video_path)) as container:
        duration = _probe_duration(container)
        stream = container.streams.video[0]

        lo = 0.0 if start_time is None else max(0.0, float(start_time))
        hi = duration if (end_time is None or end_time <= 0) else float(end_time)
        if hi <= lo:
            hi = duration if duration > lo else lo + 1.0

        span = hi - lo
        if span <= 0:
            target_times = [lo]
        else:
            step = span / (num_frames + 1)
            target_times = [lo + step * (i + 1) for i in range(num_frames)]

        for t in target_times:
            try:
                offset = int(t / stream.time_base) if stream.time_base else int(t)
                container.seek(offset, stream=stream, any_frame=False, backward=True)
                decoded = None
                for frame in container.decode(video=0):
                    decoded = frame
                    ft = float(frame.pts * stream.time_base) if frame.pts is not None else t
                    if ft >= t:
                        break
                if decoded is not None:
                    images_b64.append(_encode_frame_jpeg(decoded))
                    timestamps.append(round(t, 1))
            except Exception:  # noqa: BLE001
                continue

    return images_b64, timestamps, round(duration, 2)


_INSPECT_PROMPT = (
    "You are inspecting frames sampled from a short briefing video. The frames are "
    "given in chronological order, each labeled with its timestamp.\n\n"
    "Your job: answer the QUERY by reading the ON-SCREEN TEXT, NUMBERS, SYMBOLS, "
    "TABLES and CONFIGURATION PANELS shown in the frames. Be extremely precise:\n"
    "- Report exact numbers with their units and any thousands/万/亿 scale.\n"
    "- Report comparison operators exactly (>, >=, <, <=, =) and whether ranges "
    "are inclusive/exclusive.\n"
    "- Report any date/year/口径/batch qualifier shown alongside a threshold.\n"
    "- Cite the timestamp where you read each value.\n"
    "If the requested information is not visible in these frames, say so explicitly "
    "and suggest which time range to inspect next. Do NOT invent values.\n\n"
    "At the very END of your reply, on its own line, output a confidence score for how "
    "clearly you could read the exact value(s) the QUERY asks for, formatted EXACTLY as:\n"
    "CONFIDENCE: <a number between 0.0 and 1.0>\n"
    "Use a low score (<0.6) if digits/operators were blurry, ambiguous, partially "
    "off-screen, or not found."
)


def _parse_confidence(text: str) -> float | None:
    """Extract the trailing CONFIDENCE: <float> the inspector is asked to emit."""
    matches = re.findall(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if not matches:
        return None
    try:
        return max(0.0, min(1.0, float(matches[-1])))
    except ValueError:
        return None


def _cited_timestamps(text: str) -> list[float]:
    """Best-effort extraction of timestamps the inspector cited (e.g. '12.3s')."""
    out: list[float] = []
    for m in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*s\b", text):
        try:
            out.append(float(m))
        except ValueError:
            continue
    return out


class VideoInspector:
    """Vision-to-text sub-agent: looks at video frames, returns TEXT only.

    Inspired by VideoSeek - the main agent never holds raw video/images in its
    message history; it only sees the distilled textual observation produced here.
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # noqa: PLC0415

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=600.0,
                max_retries=2,
            )
        return self._client

    def _single_inspect(
        self,
        video_path: Path,
        *,
        query: str,
        start_time: float | None,
        end_time: float | None,
        num_frames: int,
    ) -> dict[str, Any]:
        """One vision pass over a window. Returns text + parsed confidence, no reread."""
        try:
            images, timestamps, duration = extract_frames(
                video_path,
                num_frames=num_frames,
                start_time=start_time,
                end_time=end_time,
            )
        except ImportError:
            return {"ok": False, "error": "PyAV (av) is not installed; cannot read video."}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Failed to extract frames: {exc}"}

        if not images:
            return {"ok": False, "error": "No frames could be extracted from the requested range."}

        window = (
            f"{start_time if start_time is not None else 0.0:.1f}s - "
            f"{end_time if end_time is not None else duration:.1f}s"
        )
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"{_INSPECT_PROMPT}\n\nVideo duration: {duration:.1f}s. "
                    f"Inspecting window {window} with {len(images)} frames.\n\nQUERY: {query}"
                ),
            }
        ]
        for b64, ts in zip(images, timestamps):
            content.append({"type": "text", "text": f"[{ts:.1f}s]"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Video inspection sub-call failed: {exc}"}

        return {
            "ok": True,
            "video_path": str(video_path),
            "duration_sec": duration,
            "window": window,
            "frames_inspected": len(images),
            "timestamps": [float(t) for t in timestamps],
            "observation": text,
            "confidence": _parse_confidence(text),
        }

    def inspect(
        self,
        video_path: Path,
        *,
        query: str,
        start_time: float | None = None,
        end_time: float | None = None,
        num_frames: int = 12,
    ) -> dict[str, Any]:
        if not video_path.exists():
            return {"ok": False, "error": f"Video not found: {video_path}"}

        first = self._single_inspect(
            video_path,
            query=query,
            start_time=start_time,
            end_time=end_time,
            num_frames=num_frames,
        )
        if not first.get("ok"):
            return first

        conf = first.get("confidence")
        # Auto re-read once when the read was low-confidence: narrow the window around
        # the timestamps the model cited (if any) and sample more frames for a denser read.
        if conf is not None and conf < _LOW_CONFIDENCE_THRESHOLD:
            cited = _cited_timestamps(first.get("observation", ""))
            duration = float(first.get("duration_sec") or 0.0)
            if cited:
                lo = max(0.0, min(cited) - 2.0)
                hi = (max(cited) + 2.0) if duration <= 0 else min(duration, max(cited) + 2.0)
            else:
                lo, hi = start_time, end_time
            dense_frames = min(_MAX_FRAMES, max(num_frames + 8, 16))
            second = self._single_inspect(
                video_path,
                query=query,
                start_time=lo,
                end_time=hi,
                num_frames=dense_frames,
            )
            if second.get("ok"):
                second_conf = second.get("confidence")
                best = second if (second_conf is not None and second_conf >= (conf or 0.0)) else first
                best = dict(best)
                best["reread"] = True
                best["reread_window"] = second.get("window")
                best["first_pass_confidence"] = conf
                return best

        first["reread"] = False
        return first
