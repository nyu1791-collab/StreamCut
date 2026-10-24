"""FFmpeg input handling and single-pass, frame-budgeted rendering."""

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np

from .model import CutError
from .store import command, digest
from .timeline import output_seconds, rate

SAMPLE_RATE = 16000


def tools():
    if any(not shutil.which(name) for name in ("ffmpeg", "ffprobe")):
        raise CutError("install ffmpeg and ffprobe first")


def probe(source):
    tools()
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)],
            capture_output=True,
            timeout=60,
            check=True,
        )
        data = json.loads(p.stdout)
        video = next(
            s
            for s in data["streams"]
            if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")
        )
        audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
        duration = float(data["format"]["duration"])
        fps = rate(video.get("avg_frame_rate") or video["r_frame_rate"])
        if not math.isfinite(duration) or not 1 <= duration <= 43200:
            raise ValueError
    except (KeyError, StopIteration, ValueError, subprocess.SubprocessError):
        raise CutError("input must be a finite video with audio, at most 12 hours") from None
    return {
        "duration": duration,
        "fps": str(fps),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "audio_start": float(audio.get("start_time", 0)),
        "start_time": float(data["format"].get("start_time", 0)),
        "video_duration": float(video.get("duration", duration)),
        "audio_duration": float(audio.get("duration", duration)),
    }


def acquire(source, root):
    """Download once into the shared project; never expose signed URLs in stdout."""
    destination = root / "input"
    destination.mkdir(exist_ok=True)
    if source.startswith("https://"):
        if importlib.util.find_spec("yt_dlp") is None:
            raise CutError("install the fetch extra for URL inputs")
        receipt = destination / "download-path.txt"
        command(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--no-playlist",
                "--no-progress",
                "--no-warnings",
                "--compat-options",
                "no-certifi",
                "--match-filter",
                "!is_live",
                "--restrict-filenames",
                "-N",
                "4",
                "-f",
                "bv*+ba/b",
                "--merge-output-format",
                "mkv",
                "--no-write-info-json",
                "--print-to-file",
                "after_move:filepath",
                str(receipt),
                "-o",
                str(destination / "source.%(ext)s"),
                "--",
                source,
            ],
            root / "logs" / "download.log",
        )
        if not receipt.exists():
            raise CutError("download did not produce a media path")
        paths = receipt.read_text().splitlines()
        path = Path(paths[-1]).resolve()
        if not path.is_relative_to(destination.resolve()) or not path.is_file():
            raise CutError("download returned an invalid media path")
    else:
        original = Path(source).expanduser().resolve()
        if not original.is_file():
            raise CutError("input file does not exist")
        suffix = original.suffix.lower()
        if suffix not in {".mp4", ".mkv", ".mov", ".webm", ".ts", ".flv"}:
            raise CutError("unsupported local media container")
        path = destination / ("source" + suffix)
        if path.exists() and digest(path) != digest(original):
            raise CutError("input destination already contains a different source")
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            shutil.copyfile(original, temporary)
            os.replace(temporary, path)
    return path, probe(path)


def extract_audio(source, output, log, start=0, duration=None):
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y"]
    if start:
        args += ["-ss", f"{start:.9f}"]
    args += [
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "aresample=16000:async=1:first_pts=0",
    ]
    if duration is not None:
        args += ["-t", f"{duration:.9f}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".partial.wav")
    command([*args, "-c:a", "pcm_s16le", str(temporary)], log)
    os.replace(temporary, output)


def samples(path):
    with wave.open(str(path), "rb") as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise CutError("expected 16 kHz mono PCM16 audio")
        data = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32)
    data *= 1 / 32768.0
    return data


def partition(audio, count, root):
    """Search speech-free split points. Reduce parallelism rather than cut known speech."""
    with wave.open(str(audio), "rb") as source:
        duration = source.getnframes() / SAMPLE_RATE
        cuts, warnings = [0], []
        if count > 1:
            try:
                from faster_whisper.vad import VadOptions, get_speech_timestamps
            except ImportError:
                raise CutError("parallel preparation needs the vad extra") from None
            for i in range(1, count):
                nominal = duration * i / count
                lo, hi = max(cuts[-1] + 1, nominal - 90), min(duration, nominal + 90)
                if hi - lo < 0.8:
                    warnings.append(f"split {i}: too little audio; using fewer ASR parts")
                    continue
                source.setpos(round(lo * SAMPLE_RATE))
                data = (
                    np.frombuffer(
                        source.readframes(round((hi - lo) * SAMPLE_RATE)), dtype="<i2"
                    ).astype(np.float32)
                    / 32768
                )
                voiced = get_speech_timestamps(data, VadOptions(speech_pad_ms=0))
                gaps, previous = [], 0.0
                for part in voiced + [{"start": len(data), "end": len(data)}]:
                    a, b = previous, part["start"] / SAMPLE_RATE
                    if b - a >= 0.8:
                        middle = lo + (a + b) / 2
                        if cuts[-1] + 1 < middle < duration - 1:
                            gaps.append((b - a, -abs(middle - nominal), middle))
                    previous = part["end"] / SAMPLE_RATE
                if gaps:
                    cuts.append(max(gaps)[2])
                else:
                    warnings.append(f"split {i}: no speech-free gap; using fewer ASR parts")
        cuts.append(duration)
        shards = []
        for i, (a, b) in enumerate(zip(cuts, cuts[1:], strict=False)):
            first, last = round(a * SAMPLE_RATE), round(b * SAMPLE_RATE)
            output = root / "audio" / f"part-{i:04d}.wav"
            source.setpos(first)
            with wave.open(str(output), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(SAMPLE_RATE)
                remaining = last - first
                while remaining:
                    n = min(remaining, SAMPLE_RATE * 60)
                    target.writeframes(source.readframes(n))
                    remaining -= n
            shards.append(
                {
                    "index": i,
                    "start": first / SAMPLE_RATE,
                    "end": last / SAMPLE_RATE,
                    "audio": str(output.relative_to(root)),
                    "sha256": digest(output),
                }
            )
    return shards, warnings


def silence_regions(audio, words, config):
    if not config.silence_cut:
        return []
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError:
        raise CutError("silence cutting needs the vad extra") from None
    hop = 320  # 20 ms. The AND condition preserves loud, nonverbal gameplay.
    count = len(audio) // hop
    if not count:
        return []
    rms = np.sqrt(np.mean(audio[: count * hop].reshape(count, hop) ** 2, axis=1))
    quiet = rms < 10 ** (config.silence_db / 20)
    voiced = get_speech_timestamps(audio, VadOptions(speech_pad_ms=0))
    protection = [(v["start"] / SAMPLE_RATE, v["end"] / SAMPLE_RATE) for v in voiced]
    protection += [(w["start"], w["end"]) for w in words]
    for a, b in protection:
        first = max(0, math.floor((a - config.silence_pad) * SAMPLE_RATE / hop))
        last = min(count, math.ceil((b + config.silence_pad) * SAMPLE_RATE / hop))
        quiet[first:last] = False
    changes = np.diff(np.concatenate(([False], quiet, [False])).astype(np.int8))
    cuts = []
    for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True):
        x, y = a * hop / SAMPLE_RATE, b * hop / SAMPLE_RATE
        if y - x >= config.silence_gap:
            x, y = x + config.silence_pad, y - config.silence_pad
            if y > x:
                cuts.append((x, y))
    return cuts


def nvenc_available():
    try:
        p = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=s=320x240:r=30",
                "-frames:v",
                "1",
                "-an",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=20,
        )
        return p.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def render(source, directory, plan, config, log):
    fps = Fraction(plan["fps"])
    origin = float(Fraction(plan["first"]) / fps)
    length = float(Fraction(plan["last"] - plan["first"]) / fps)
    keep = [(a - plan["first"], b - plan["first"]) for a, b in plan["keep_frames"]]
    n = len(keep)
    chain = [
        f"[0:v:0]fps={fps},setpts=PTS-STARTPTS,format=yuv420p,"
        f"split={n}" + "".join(f"[vs{i}]" for i in range(n)),
        f"[0:a:0]aresample=48000:async=1:first_pts=0,apad,asplit={n}"
        + "".join(f"[as{i}]" for i in range(n)),
    ]
    for i, (a, b) in enumerate(keep):
        x, y = float(Fraction(a) / fps), float(Fraction(b) / fps)
        # Audio pieces are forced to the same rational frame durations as video.
        sample_a, sample_b = round(Fraction(a) * 48000 / fps), round(Fraction(b) * 48000 / fps)
        fade = min(0.005, (y - x) / 2)
        chain += [
            f"[vs{i}]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[v{i}]",
            f"[as{i}]atrim=start_sample={sample_a}:end_sample={sample_b},"
            f"asetpts=PTS-STARTPTS,afade=t=in:d={fade:.9f},"
            f"afade=t=out:st={y - x - fade:.9f}:d={fade:.9f}[a{i}]",
        ]
    labels = "".join(f"[v{i}][a{i}]" for i in range(n))
    chain += [
        f"{labels}concat=n={n}:v=1:a=1[vc][ac]",
        "[vc]" + ("ass=captions.ass" if config.burn_in else "null") + "[vout]",
        "[ac]alimiter=limit=0.707946:level=0[aout]",
    ]
    graph = directory / "render.ffgraph"
    graph.write_text(";\n".join(chain), encoding="utf-8")
    encoder = config.encoder
    if encoder == "auto":
        encoder = "h264_nvenc" if nvenc_available() else "libx264"
    elif encoder == "h264_nvenc" and not nvenc_available():
        raise CutError("requested NVENC encoder is not usable on this runtime")
    options = (
        ["-preset", "p4", "-rc", "vbr", "-cq", "20", "-b:v", "0"]
        if encoder == "h264_nvenc"
        else ["-preset", "veryfast", "-crf", "18"]
    )
    threads = config.ffmpeg_threads or available_cpus()
    output = directory / "video.mp4"
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-threads",
            str(threads),
            "-ss",
            f"{origin:.9f}",
            "-i",
            str(source.resolve()),
            "-t",
            f"{length:.9f}",
            "-filter_complex_threads",
            str(threads),
            "-filter_complex_script",
            str(graph.resolve()),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            encoder,
            *options,
            "-threads",
            str(threads),
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-fps_mode",
            "cfr",
            "-frames:v",
            str(plan["frames"]),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-map_metadata",
            "-1",
            str(output.resolve()),
        ],
        log,
        cwd=directory,
    )
    expected = output_seconds(plan)
    actual = probe(output)
    if abs(actual["duration"] - expected) > max(0.1, 2 / float(fps)):
        raise CutError("rendered duration differs from the edit plan")
    if abs(actual["audio_duration"] - actual["video_duration"]) > max(0.05, 1 / float(fps)):
        raise CutError("rendered audio and video durations differ")
    command(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(output.resolve()),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        log.with_name(log.stem + "-decode.log"),
    )
    return {
        "encoder": encoder,
        "frames": plan["frames"],
        "duration": actual["duration"],
        "expected_duration": expected,
        "decode_checked": True,
    }


def available_cpus():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def local_audio(path):
    """Stage only a small WAV off Drive before inference."""
    temporary = tempfile.TemporaryDirectory(prefix="streamcut-")
    target = Path(temporary.name) / "audio.wav"
    shutil.copyfile(path, target)
    return temporary, target
