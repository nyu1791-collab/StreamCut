"""Lazy WhisperX models: recognise once, align only selected audio."""

import gc
import unicodedata

from .media import available_cpus, local_audio, samples
from .model import CutError, span, text
from .store import digest, within


def compact(value):
    return "".join(unicodedata.normalize("NFC", value).split())


class Models:
    def __init__(self, config):
        self.config = config
        self.asr = None
        self.aligner = None
        self.metadata = None

    def backend(self):
        try:
            import torch
            import whisperx
        except ImportError:
            raise CutError("install the gpu extra on a CUDA runtime") from None
        if not torch.cuda.is_available():
            raise CutError("this inference worker requires a CUDA GPU")
        torch.set_num_threads(available_cpus())
        return torch, whisperx

    def release_asr(self):
        if self.asr is not None:
            self.asr = None
            gc.collect()
            torch, _ = self.backend()
            torch.cuda.empty_cache()

    def close(self):
        loaded = self.asr is not None or self.aligner is not None
        self.asr = self.aligner = self.metadata = None
        if loaded:
            gc.collect()
            torch, _ = self.backend()
            torch.cuda.empty_cache()

    def transcribe(self, root, payload):
        path = within(root, payload["audio"])
        if digest(path) != payload["sha256"]:
            raise CutError("ASR audio hash mismatch")
        _, whisperx = self.backend()
        if self.asr is None:
            self.asr = whisperx.load_model(
                self.config.asr_model,
                "cuda",
                compute_type="float16",
                language=self.config.language,
                vad_method="silero",
                threads=available_cpus(),
            )
        temporary, local = local_audio(path)
        try:
            data = samples(local)
            result = self.asr.transcribe(
                data, batch_size=self.config.batch_size, language=self.config.language
            )
        finally:
            temporary.cleanup()
        units = []
        duration = payload["end"] - payload["start"]
        for row in result["segments"]:
            a, b = max(0, float(row["start"])), min(duration, float(row["end"]))
            if b <= a or not row["text"].strip():
                continue
            units.append(
                {
                    "start": a + payload["start"],
                    "end": b + payload["start"],
                    "text": text(row["text"]),
                    "words": [],
                }
            )
        return {
            "units": units,
            "language": result.get("language", self.config.language),
            "asr_model": self.config.asr_model,
        }

    def align(self, root, payload):
        self.release_asr()
        _, whisperx = self.backend()
        if self.aligner is None:
            self.aligner, self.metadata = whisperx.load_align_model(
                language_code=self.config.language, device="cuda"
            )
        path = within(root, payload["audio"])
        if digest(path) != payload["sha256"]:
            raise CutError("alignment audio hash mismatch")
        temporary, local = local_audio(path)
        try:
            audio = samples(local)
            origin = payload["origin"]
            segments = [
                {
                    "start": max(0, u["start"] - origin),
                    "end": min(len(audio) / 16000, u["end"] - origin),
                    "text": u["text"],
                }
                for u in payload["units"]
            ]
            result = whisperx.align(
                segments, self.aligner, self.metadata, audio, "cuda", return_char_alignments=True
            )
        finally:
            temporary.cleanup()
        if compact("".join(s["text"] for s in result["segments"])) != compact(
            "".join(s["text"] for s in segments)
        ):
            raise CutError("alignment changed or lost transcript text")
        words, estimated = alignment_words(result["segments"], len(audio) / 16000)
        return {
            "words": [{**w, "start": w["start"] + origin, "end": w["end"] + origin} for w in words],
            "estimated_characters": estimated,
            "alignment": "whisperx",
        }


def alignment_words(segments, duration):
    """Keep unsupported numbers/names; mark estimated timings instead of deleting text."""
    output, estimated = [], 0
    for segment in segments:
        a, b = span(float(segment["start"]), float(segment["end"]), duration)
        original = unicodedata.normalize("NFC", text(segment["text"]))
        chars = segment.get("chars", [])
        if compact("".join(c.get("char", "") for c in chars)) != compact(original):
            chars = [{"char": c} for c in original]
        # Invalid/missing timings are bounded by neighbouring valid characters, when possible.
        valid = []
        for i, c in enumerate(chars):
            x, y = c.get("start"), c.get("end")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and a <= x < y <= b:
                valid.append((i, float(x), float(y)))
        anchors = {i: (x, y) for i, x, y in valid}
        for i, c in enumerate(chars):
            value = c.get("char", "")
            if not value:
                continue
            approximated = i not in anchors
            if approximated:
                previous = next(((j, y) for j, _, y in reversed(valid) if j < i), (-1, a))
                following = next(((j, x) for j, x, _ in valid if j > i), (len(chars), b))
                lo, hi = previous[1], max(previous[1], following[1])
                width = (hi - lo) / (following[0] - previous[0] - 1)
                x = lo + (i - previous[0] - 1) * width
                y = x + width
                if y <= x:
                    # Preserve text at a tiny interval within the parent utterance.
                    x, y = max(a, min(lo, b - 0.001)), min(b, max(lo + 0.001, a + 0.001))
                estimated += int(not value.isspace())
            else:
                x, y = anchors[i]
            output.append({"start": x, "end": y, "text": value, "estimated": approximated})
    if not output:
        raise CutError("alignment returned no characters")
    return output, estimated
