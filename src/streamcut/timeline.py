"""One frame grid and one keep list for video, audio and captions."""

import math
from fractions import Fraction

from .model import CutError, span


def rate(value):
    try:
        fps = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        raise CutError("invalid frame rate") from None
    if not 1 <= fps <= 240:
        raise CutError("unsupported source frame rate")
    return min(fps, Fraction(30))


def merge(regions):
    output = []
    for a, b in sorted(regions):
        if b <= a:
            continue
        if output and a <= output[-1][1]:
            output[-1] = (output[-1][0], max(output[-1][1], b))
        else:
            output.append((a, b))
    return output


def frame_plan(start, end, source_duration, fps, removed=()):
    """Round the outer clip outwards and removed gaps inwards, preserving speech."""
    fps = rate(fps)
    span(start, end, source_duration)
    first = math.floor(Fraction(str(start)) * fps)
    last = min(
        math.ceil(Fraction(str(end)) * fps), math.floor(Fraction(str(source_duration)) * fps)
    )
    cuts = []
    for a, b in removed:
        a = max(first, math.ceil(Fraction(str(a)) * fps))
        b = min(last, math.floor(Fraction(str(b)) * fps))
        if b > a:
            cuts.append((a, b))
    keep, cursor = [], first
    for a, b in merge(cuts):
        if cursor < a:
            keep.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < last:
        keep.append((cursor, last))
    if not keep:
        raise CutError("edit would remove the entire clip")
    return {
        "fps": str(fps),
        "first": first,
        "last": last,
        "keep_frames": keep,
        "frames": sum(b - a for a, b in keep),
    }


def output_seconds(plan):
    return float(Fraction(plan["frames"]) / Fraction(plan["fps"]))


def map_words(words, plan):
    """Map from the source clock. A speech word must never straddle a removed gap."""
    fps = Fraction(plan["fps"])
    keep = [(float(Fraction(a) / fps), float(Fraction(b) / fps)) for a, b in plan["keep_frames"]]
    output = []
    for word in words:
        matches, elapsed = [], 0.0
        for a, b in keep:
            x, y = max(a, word["start"]), min(b, word["end"])
            if y > x:
                matches.append({**word, "start": elapsed + x - a, "end": elapsed + y - a})
            elapsed += b - a
        if len(matches) > 1:
            raise CutError("a word crosses a removed interval; keep more speech padding")
        if not matches and word["text"].strip():
            raise CutError("a word falls outside the retained frames; inspect clip boundaries")
        output.extend(matches)
    return output
