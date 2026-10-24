"""Japanese word boundaries, measured line widths, ASS and SRT from one event list."""

import json
import math
import re
import subprocess
import unicodedata
from functools import lru_cache

from fontTools.ttLib import TTFont
from PIL import ImageFont

from .model import CutError


@lru_cache
def font_path(family):
    try:
        p = subprocess.run(
            ["fc-match", "--format", "%{file}\n%{index}", family],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        path, index = p.stdout.splitlines()[:2]
        return path, int(index) & 0xFFFF
    except (OSError, ValueError, subprocess.SubprocessError):
        raise CutError("install fontconfig and a Japanese font") from None


def ensure_glyphs(font, index, value):
    with TTFont(font, fontNumber=index) as face:
        known = set().union(*(table.cmap for table in face["cmap"].tables if table.isUnicode()))
    if any(ord(c) not in known for c in value if not c.isspace()):
        raise CutError("selected font cannot render all caption/headline characters")


def escape(value):
    # ASS override syntax must not be interpreted as instructions from transcript text.
    return value.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", r"\N")


def timestamp(seconds, ass=False):
    scale = 100 if ass else 1000
    total = max(0, round(seconds * scale))
    whole, fraction = divmod(total, scale)
    hours, remaining = divmod(whole, 3600)
    minutes, secs = divmod(remaining, 60)
    return (
        f"{hours}:{minutes:02d}:{secs:02d}.{fraction:02d}"
        if ass
        else f"{hours:02d}:{minutes:02d}:{secs:02d},{fraction:03d}"
    )


def lexical_tokens(words, language):
    value, bounds = "", []
    for w in words:
        normalized = unicodedata.normalize("NFC", w["text"])
        for i, c in enumerate(normalized):
            a = w["start"] + (w["end"] - w["start"]) * i / len(normalized)
            b = w["start"] + (w["end"] - w["start"]) * (i + 1) / len(normalized)
            value += c
            bounds.append((a, b))
    if not value.strip():
        raise CutError("no subtitle text survived the edit")
    if language == "ja":
        import fugashi

        surfaces = [token.surface for token in fugashi.Tagger()(value)]
    else:
        surfaces = re.findall(r"\w+(?:[-’']\w+)*|[^\w\s]", value)
    tokens, cursor = [], 0
    for surface in surfaces:
        at = value.find(surface, cursor)
        if at < cursor:
            raise CutError("subtitle tokenizer changed the transcript")
        if value[cursor:at].strip():
            raise CutError("subtitle tokenizer omitted text")
        end = at + len(surface)
        prefix = value[cursor:at] if tokens else ""
        tokens.append({"text": prefix + surface, "start": bounds[at][0], "end": bounds[end - 1][1]})
        cursor = end
    if value[cursor:].strip():
        raise CutError("subtitle tokenizer omitted trailing text")
    return tokens


def wrap(tokens, font, width):
    """Never split a morphological token; prefer balanced two-line layouts."""
    whole = "".join(t["text"] for t in tokens).strip()
    if font.getlength(whole) <= width:
        return whole
    choices = []
    for i in range(1, len(tokens)):
        a = "".join(t["text"] for t in tokens[:i]).strip()
        b = "".join(t["text"] for t in tokens[i:]).strip()
        x, y = font.getlength(a), font.getlength(b)
        if max(x, y) <= width and not b.startswith(("、", "。", "！", "？", "」", "）")):
            choices.append((abs(x - y), a + "\n" + b))
    return min(choices)[1] if choices else None


def events(words, duration, config, width, height):
    path, index = font_path(config.font)
    size = max(16, round(config.font_size * height / 1080))
    font = ImageFont.truetype(path, size, index=index)
    line_width = width * 0.90
    tokens = lexical_tokens(words, config.language)
    grouped, current = [], []
    for token in tokens:
        candidate = current + [token]
        gap = current and token["start"] - current[-1]["end"] >= 0.7
        too_long = current and token["end"] - current[0]["start"] > config.max_caption_seconds
        if current and (gap or too_long or wrap(candidate, font, line_width) is None):
            grouped.append(current)
            current = []
        current.append(token)
        if wrap(current, font, line_width) is None:
            raise CutError("a subtitle word exceeds two lines; choose a smaller font")
    if current:
        grouped.append(current)
    output = []
    warnings = {
        "short_captions": 0,
        "fast_captions": 0,
        "estimated_characters": sum(len(w["text"].strip()) for w in words if w.get("estimated")),
    }
    for i, group in enumerate(grouped):
        start = max(0, group[0]["start"])
        end = min(duration, max(t["end"] for t in group))
        next_start = grouped[i + 1][0]["start"] if i + 1 < len(grouped) else duration
        if end > next_start:
            end = next_start
        content = wrap(group, font, line_width)
        count = len("".join(content.split()))
        end = min(
            duration,
            next_start,
            max(
                end,
                start + min(1, config.max_caption_seconds),
                start + min(config.max_caption_seconds, count / config.target_cps),
            ),
        )
        # Both ASS (centiseconds) and SRT use these exact quantised boundaries.
        start, end = math.ceil(start * 100) / 100, math.floor(end * 100) / 100
        if end <= start:
            raise CutError("caption has no display duration; inspect alignment")
        output.append({"start": start, "end": end, "text": content})
        warnings["short_captions"] += end - start < 0.5
        warnings["fast_captions"] += count / (end - start) > config.target_cps
    return output, warnings, (path, index, size)


def create(words, directory, duration, headline, config, width, height):
    rows, warnings, (path, index, size) = events(words, duration, config, width, height)
    ensure_glyphs(path, index, headline + "".join(r["text"] for r in rows))
    family = config.font.replace(",", " ").replace("\n", " ")
    heading_size = round(size * 1.1)
    while heading_size > 16:
        f = ImageFont.truetype(path, heading_size, index=index)
        if f.getlength(headline) <= width * 0.9:
            break
        heading_size -= 1
    else:
        raise CutError("headline does not fit; shorten it")
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
"""
    header += (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,{family},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,"
        f"-1,0,0,0,100,100,0,0,1,3,0,2,{round(width * 0.05)},{round(width * 0.05)},"
        f"{round(height * 0.12)},1\n"
        f"Style: Heading,{family},{heading_size},&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,"
        f"-1,0,0,0,100,100,0,0,3,7,0,7,{round(width * 0.04)},{round(width * 0.04)},"
        f"{round(height * 0.04)},1\n"
    )
    header += """

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass = [
        header,
        f"Dialogue: 1,0:00:00.00,{timestamp(duration, True)},Heading,,0,0,0,,"
        + escape(headline)
        + "\n",
    ]
    srt = []
    for i, row in enumerate(rows, 1):
        ass.append(
            f"Dialogue: 0,{timestamp(row['start'], True)},"
            f"{timestamp(row['end'], True)},Caption,,0,0,0,,{escape(row['text'])}\n"
        )
        srt.append(f"{i}\n{timestamp(row['start'])} --> {timestamp(row['end'])}\n{row['text']}\n\n")
    (directory / "captions.ass").write_text("".join(ass), encoding="utf-8")
    (directory / "captions.srt").write_text("".join(srt), encoding="utf-8")
    (directory / "captions.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"events": len(rows), "warnings": warnings, "font_size": size, "human_reviewed": False}
