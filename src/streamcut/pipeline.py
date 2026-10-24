"""Three phases on one GPU runtime or statically assigned Colab runtimes."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from . import __version__, media, planner
from .inference import Models
from .model import Config, CutError, text, validate_picks, validate_units
from .store import cached, code_identity, digest, fingerprint, read, task, wait, within, write
from .timeline import frame_plan
from .worker import execute


def dispatch(root, specs, config, distributed, models=None):
    if not specs:
        return []
    if not distributed or specs[0]["stage"] == "render" and config.render_workers == 1:
        return [execute(root, s, config, models) for s in specs]
    if specs[0]["stage"] == "render":
        # The coordinator is CPU rank 0: four CPU runtimes include the coordinator.
        own = [s for s in specs if s["index"] % config.render_workers == 0]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: [execute(root, s, config) for s in own])
            result = wait(root, specs, config.worker_timeout)
            future.result()
            return result
    return wait(root, specs, config.worker_timeout)


def prepare(root, source, config):
    record = root / "prepared.json"
    if record.exists():
        value = read(record)
        if digest(within(root, value["source"])) != value["source_sha256"]:
            raise CutError("cached source video has changed")
        if digest(within(root, value["audio"])) != value["audio_sha256"]:
            raise CutError("cached master audio has changed")
        return value
    print("① 取得・解析: 入力を共有フォルダに準備", flush=True)
    path, info = media.acquire(source, root)
    audio = root / "audio" / "master.wav"
    media.extract_audio(path, audio, root / "logs" / "audio.log", duration=info["duration"])
    shards, warnings = media.partition(audio, config.gpu_workers, root)
    value = {
        "source": str(path.relative_to(root)),
        "source_sha256": digest(path),
        "info": info,
        "audio": str(audio.relative_to(root)),
        "audio_sha256": digest(audio),
        "shards": shards,
        "warnings": warnings,
    }
    write(record, value)
    return value


def source_identity(source):
    if source.startswith("https://"):
        return fingerprint({"url": source})
    path = Path(source).expanduser().resolve()
    info = path.stat()
    return fingerprint({"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns})


def run(
    source, project, config, *, distributed=False, resume=False, transcript=None, selection=None
):
    os.umask(0o077)
    root = Path(project).expanduser().resolve()
    if root == Path(source).expanduser().absolute() or (root / ".git").exists():
        raise CutError("choose a separate project folder, not the input or repository")
    identity = fingerprint(
        {
            "source": source_identity(source),
            "config": config.dump(),
            "code": code_identity(),
            "version": __version__,
            "transcript": digest(transcript) if transcript else None,
            "selection": digest(selection) if selection else None,
        }
    )
    record = root / "project.json"
    if record.exists():
        job = read(record)
        if job.get("identity") != identity:
            raise CutError("input/config/code changed; choose a new project directory")
        if not resume:
            raise CutError("project exists; use --resume for the same input and settings")
        if job["state"] == "complete":
            # A completion flag alone is insufficient after a Drive file was changed or deleted.
            prepared = read(root / "prepared.json")
            if digest(within(root, prepared["source"])) != prepared["source_sha256"]:
                raise CutError("completed project source was modified")
            specs = [read(p) for p in sorted((root / "tasks/render").glob("*.json"))]
            if len(specs) != len(job["clips"]) or any(cached(root, s) is None for s in specs):
                raise CutError("completed project is missing verified clip results")
            print("完了済み: 保存された素材を再利用します", flush=True)
            return job
        # Explicit resume retries failed tasks, while retaining successful task receipts.
        for path in (root / "results").glob("*/*.json"):
            if read(path).get("error"):
                path.unlink()
    else:
        if root.exists() and any(root.iterdir()):
            raise CutError("new project directory must be empty")
        root.mkdir(parents=True, exist_ok=True)
        (root / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
        job = {
            "schema": 1,
            "identity": identity,
            "config": config.dump(),
            "timings": {},
            "state": "preparing",
            "human_reviewed": False,
        }
    lease = root / ".coordinator.lock"
    try:
        lock = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise CutError(
            "coordinator lock exists; confirm the old coordinator stopped before removing it"
        ) from None
    os.close(lock)
    started = time.monotonic()
    write(record, job)

    def state(name):
        job["state"] = name
        write(record, job)

    def timed(name, function):
        begin = time.monotonic()
        value = function()
        job["timings"][name] = round(time.monotonic() - begin, 3)
        return value

    try:
        if not selection and not (root / "selection.json").exists():
            if not config.llm_model or not os.environ.get(config.llm_key_env):
                raise CutError("set the LLM model and API key, or supply --selection")
        prepared = timed("prepare", lambda: prepare(root, source, config))
        info = prepared["info"]
        models = None if distributed else Models(config)
        if transcript:
            value = read(transcript)
            units = validate_units(value["units"], info["duration"])
        else:
            specs = [
                task(
                    root,
                    "asr",
                    i,
                    {
                        **s,
                        "model": config.asr_model,
                        "batch_size": config.batch_size,
                        "language": config.language,
                        "code": code_identity(),
                    },
                )
                for i, s in enumerate(prepared["shards"])
            ]
            state("transcribing")
            results = timed("asr", lambda: dispatch(root, specs, config, distributed, models))
            units = validate_units(
                [u for result in results for u in result["units"]], info["duration"]
            )
        write(root / "transcript.json", {"units": units, "language": config.language})
        state("planning")
        print("② 話題・採用区間・見出しを選定", flush=True)
        selected_path = root / "selection.json"
        if selected_path.exists():
            chosen = read(selected_path)
        elif selection:
            chosen = read(selection)
        else:
            chosen = timed("planning", lambda: planner.plan(root, units, config, resume))
        clips = validate_picks(chosen, units, config)
        write(selected_path, chosen)
        print("③ 採用区間の時刻精密化・編集・出力", flush=True)
        align_specs, clip_audio, supplied = [], [], []
        for clip in clips:
            frames = frame_plan(clip["start"], clip["end"], info["duration"], info["fps"])
            fps = Fraction(frames["fps"])
            origin = float(Fraction(frames["first"]) / fps)
            duration = float(Fraction(frames["last"] - frames["first"]) / fps)
            audio = root / "audio" / f"clip-{clip['index']:04d}.wav"
            if not audio.exists():
                media.extract_audio(
                    within(root, prepared["source"]),
                    audio,
                    root / "logs" / f"audio-clip-{clip['index']:04d}.log",
                    origin,
                    duration,
                )
            clip_audio.append((audio, origin))
            if config.alignment == "provided":
                words = [w for u in clip["units"] for w in u["words"]]
                if any(not u["words"] for u in clip["units"]):
                    raise CutError("provided alignment requires words for every selected unit")
                supplied.append(
                    {"words": words, "estimated_characters": 0, "alignment": "provided"}
                )
            else:
                align_specs.append(
                    task(
                        root,
                        "align",
                        clip["index"],
                        {
                            "audio": str(audio.relative_to(root)),
                            "sha256": digest(audio),
                            "origin": origin,
                            "units": clip["units"],
                            "language": config.language,
                            "code": code_identity(),
                        },
                    )
                )
        state("aligning")
        aligned = (
            supplied
            if config.alignment == "provided"
            else timed(
                "alignment", lambda: dispatch(root, align_specs, config, distributed, models)
            )
        )
        if models:
            models.close()
        specs = [
            task(
                root,
                "render",
                clip["index"],
                {
                    "source": prepared["source"],
                    "source_sha256": prepared["source_sha256"],
                    "info": info,
                    "clip": clip,
                    "audio": str(audio.relative_to(root)),
                    "audio_sha256": digest(audio),
                    "origin": origin,
                    "aligned": alignment,
                    "config": config.dump(),
                    "code": code_identity(),
                },
            )
            for clip, alignment, (audio, origin) in zip(clips, aligned, clip_audio, strict=True)
        ]
        state("rendering")
        results = timed("render", lambda: dispatch(root, specs, config, distributed))
        job["clips"] = [
            {
                "directory": f"clips/{i + 1:02d}",
                "headline": c["headline"],
                "start": c["start"],
                "end": c["end"],
                "result": result,
            }
            for i, (c, result) in enumerate(zip(clips, results, strict=True))
        ]
        job["warnings"] = prepared["warnings"]
        job["timings"]["total"] = round(time.monotonic() - started, 3)
        state("complete")
        print(f"完了: {len(clips)} 本。内容・字幕は未確認の下書きです。", flush=True)
        return job
    except Exception as error:
        job["error_type"] = type(error).__name__
        state("failed")
        raise
    finally:
        lease.unlink(missing_ok=True)


def reexport(project, overrides=None, headlines=None):
    """Reuse selected alignment; keep original outputs and ASR/LLM receipts untouched."""
    root = Path(project).expanduser().resolve()
    job = read(root / "project.json")
    if job["state"] != "complete":
        raise CutError("finish or resume the original project before exporting adjustments")
    allowed = {
        "silence_cut",
        "silence_gap",
        "silence_pad",
        "silence_db",
        "encoder",
        "burn_in",
        "font",
        "font_size",
        "max_caption_seconds",
        "target_cps",
        "ffmpeg_threads",
    }
    overrides, headlines = overrides or {}, headlines or {}
    if not isinstance(overrides, dict) or set(overrides) - allowed:
        raise CutError("export accepts caption/silence/encoding settings only")
    if not isinstance(headlines, dict) or set(headlines) - {
        f"{i + 1:02d}" for i in range(len(job["clips"]))
    }:
        raise CutError("headlines must map clip numbers such as 01 to new text")
    headlines = {k: text(v, "headline", 100) for k, v in headlines.items()}
    if any("\n" in v or "\t" in v for v in headlines.values()):
        raise CutError("headline must fit on one line")
    config = Config.load({**job["config"], **overrides})
    prepared = read(root / "prepared.json")
    if digest(within(root, prepared["source"])) != prepared["source_sha256"]:
        raise CutError("cached source video has changed")
    specs = [read(p) for p in sorted((root / "tasks/render").glob("*.json"))]
    if len(specs) != len(job["clips"]):
        raise CutError("original render specifications are missing")
    identity = fingerprint(
        {
            "config": config.dump(),
            "headlines": headlines,
            "source": prepared["source_sha256"],
            "code": code_identity(),
        }
    )[:16]
    output = root / "exports" / identity
    output.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if cached(root, spec) is None:
            raise CutError("original render receipt is missing")
        payload = dict(spec["payload"])
        payload["clip"] = dict(payload["clip"])
        index = payload["clip"]["index"]
        if f"{index + 1:02d}" in headlines:
            payload["clip"]["headline"] = headlines[f"{index + 1:02d}"]
        payload.update(
            config=config.dump(), code=code_identity(), output_subdir=f"exports/{identity}/clips"
        )
        changed = task(root, "export", index, payload)
        execute(root, changed, config)
    print(f"再書き出し完了: exports/{identity}/clips。ASR・LLMの再実行なし。", flush=True)
    return output
