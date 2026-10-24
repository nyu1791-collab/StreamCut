"""Small, explicit records shared by the coordinator and workers."""

import math
import re
from dataclasses import asdict, dataclass, fields


class CutError(RuntimeError):
    """A bounded, user-facing error. Never attach provider responses or secrets."""


def number(value, name="time"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CutError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise CutError(f"{name} must be a finite number")
    return float(value)


def span(start, end, limit=math.inf):
    start, end = number(start), number(end)
    if not 0 <= start < end <= limit + 0.001:
        raise CutError("time range is outside its source")
    return start, min(end, limit)


def text(value, name="text", limit=16000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise CutError(f"invalid {name}")
    if any(ord(c) < 32 and c not in "\n\t" for c in value):
        raise CutError(f"invalid control character in {name}")
    return value.strip()


@dataclass(frozen=True)
class Config:
    language: str = "ja"
    asr_model: str = "turbo"
    batch_size: int = 8
    gpu_workers: int = 1
    render_workers: int = 1
    clips: int = 3
    min_seconds: float = 20
    max_seconds: float = 600
    silence_cut: bool = True
    silence_gap: float = 1.5
    silence_pad: float = 0.2
    silence_db: float = -40
    encoder: str = "auto"
    burn_in: bool = True
    font: str = "Noto Sans CJK JP"
    font_size: int = 52
    max_caption_seconds: float = 4
    target_cps: float = 7
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = ""
    llm_key_env: str = "STREAMCUT_API_KEY"
    plan_chars: int = 28000
    worker_timeout: int = 7200
    ffmpeg_threads: int = 0
    alignment: str = "whisperx"

    @classmethod
    def load(cls, data):
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise CutError("unknown configuration field")
        config = cls(**data)
        for name, low, high in (
            ("batch_size", 1, 32),
            ("gpu_workers", 1, 4),
            ("render_workers", 1, 4),
            ("clips", 1, 20),
            ("font_size", 18, 120),
            ("plan_chars", 2000, 100000),
            ("worker_timeout", 30, 86400),
            ("ffmpeg_threads", 0, 1024),
        ):
            value = getattr(config, name)
            if type(value) is not int or not low <= value <= high:
                raise CutError(f"invalid configuration: {name}")
        for name in ("silence_cut", "burn_in"):
            if type(getattr(config, name)) is not bool:
                raise CutError(f"invalid configuration: {name}")
        for name in (
            "min_seconds",
            "max_seconds",
            "silence_gap",
            "silence_pad",
            "silence_db",
            "max_caption_seconds",
            "target_cps",
        ):
            number(getattr(config, name), name)
        if not 1 <= config.min_seconds < config.max_seconds <= 3600:
            raise CutError("clip length must be between 1 second and 1 hour")
        if not 0 <= config.silence_pad <= 1 or not 0.5 <= config.silence_gap <= 10:
            raise CutError("invalid silence settings")
        if not -90 <= config.silence_db <= -10:
            raise CutError("invalid silence threshold")
        if not 0.5 <= config.max_caption_seconds <= 10 or not 1 <= config.target_cps <= 30:
            raise CutError("invalid caption settings")
        if config.encoder not in {"auto", "libx264", "h264_nvenc"}:
            raise CutError("invalid video encoder")
        if config.alignment not in {"whisperx", "provided"}:
            raise CutError("invalid alignment mode")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config.llm_key_env):
            raise CutError("invalid API key environment variable name")
        for name in ("language", "asr_model", "font"):
            text(getattr(config, name), name, 200)
        if not isinstance(config.llm_model, str) or len(config.llm_model) > 200:
            raise CutError("invalid LLM model ID")
        if not isinstance(config.llm_base_url, str):
            raise CutError("invalid LLM base URL")
        return config

    def dump(self):
        return asdict(self)


def validate_units(rows, duration):
    if not isinstance(rows, list) or not rows:
        raise CutError("transcript must contain at least one unit")
    output = []
    last_end = 0
    for i, row in enumerate(rows, 1):
        a, b = span(row.get("start"), row.get("end"), duration)
        if a < last_end - 0.05:
            raise CutError("transcript units overlap or are out of order")
        words = []
        previous = a
        for w in row.get("words", []):
            x, y = span(w.get("start"), w.get("end"), b)
            if x < a - 0.001 or x < previous - 0.05:
                raise CutError("word timing is outside its unit or out of order")
            words.append({"start": x, "end": y, "text": text(w.get("text"))})
            previous = y
        output.append(
            {"id": f"u{i:06d}", "start": a, "end": b, "text": text(row.get("text")), "words": words}
        )
        last_end = b
    return output


def validate_picks(value, units, config):
    if not isinstance(value, dict) or not isinstance(value.get("clips"), list):
        raise CutError("selection must contain clips")
    if not 1 <= len(value["clips"]) <= config.clips:
        raise CutError("selection clip count is outside the requested range")
    positions = {u["id"]: i for i, u in enumerate(units)}
    picks = []
    for i, row in enumerate(value["clips"]):
        if not isinstance(row, dict):
            raise CutError("invalid selection row")
        a, b = positions.get(row.get("start_id")), positions.get(row.get("end_id"))
        if a is None or b is None or a > b:
            raise CutError("selection references an unknown or reversed unit ID")
        start, end = units[a]["start"], units[b]["end"]
        if not config.min_seconds <= end - start <= config.max_seconds:
            raise CutError("selected clip length is outside the configured range")
        if any(start < p["end"] and end > p["start"] for p in picks):
            raise CutError("selected clips overlap")
        headline = text(row.get("headline"), "headline", 100)
        if "\n" in headline or "\t" in headline:
            raise CutError("headline must fit on one line")
        picks.append(
            {
                "index": i,
                "start": start,
                "end": end,
                "headline": headline,
                "start_id": units[a]["id"],
                "end_id": units[b]["id"],
                "units": units[a : b + 1],
            }
        )
    return picks
