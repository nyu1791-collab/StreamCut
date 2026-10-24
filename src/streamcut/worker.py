"""Statically partitioned Colab workers; no shared job-claim database or lock races."""

import os
import tempfile
import time
from pathlib import Path

from . import captions, media
from .inference import Models
from .model import Config, CutError
from .store import (
    cached,
    code_identity,
    digest,
    fingerprint,
    read,
    result_path,
    save_result,
    within,
    write,
)
from .timeline import frame_plan, map_words, output_seconds


def validate_task(spec, stage):
    body = {k: spec[k] for k in ("schema", "stage", "index", "payload")}
    if body["stage"] != stage or spec["key"] != fingerprint(body):
        raise CutError("task identity mismatch")


def execute(root, spec, config, models=None):
    stage = spec["stage"]
    validate_task(spec, stage)
    if spec["payload"].get("code") != code_identity():
        raise CutError("worker code differs from the coordinator; install the same revision")
    previous = cached(root, spec)
    if previous is not None:
        return previous
    started = time.monotonic()
    payload = spec["payload"]
    try:
        if stage in {"asr", "align"}:
            models = models or Models(config)
            value = (
                models.transcribe(root, payload) if stage == "asr" else models.align(root, payload)
            )
            save_result(root, spec, value, elapsed=time.monotonic() - started)
        elif stage in {"render", "export"}:
            value, files = render_clip(root, payload, config)
            save_result(root, spec, value, files, time.monotonic() - started)
        else:
            raise CutError("unknown task stage")
        return value
    except Exception as error:
        # Bound diagnostics; SDK exceptions can carry private transcript text and signed URLs.
        write(
            result_path(root, spec),
            {"schema": 1, "key": spec["key"], "error": type(error).__name__},
        )
        raise


def render_clip(root, payload, config):
    aligned = payload["aligned"]
    source = within(root, payload["source"])
    audio_path = within(root, payload["audio"])
    if digest(audio_path) != payload["audio_sha256"]:
        raise CutError("render audio was modified after alignment")
    clip = payload["clip"]
    origin = payload["origin"]
    words = aligned["words"]
    local_words = [{**w, "start": w["start"] - origin, "end": w["end"] - origin} for w in words]
    audio = media.samples(audio_path)
    removed = media.silence_regions(audio, local_words, config)
    removed = [(a + origin, b + origin) for a, b in removed]
    info = payload["info"]
    timeline = frame_plan(clip["start"], clip["end"], info["duration"], info["fps"], removed)
    shifted = map_words(words, timeline)
    # Temporary outputs live beside the destination, so final rename stays on one filesystem.
    parent = within(root, payload.get("output_subdir", "clips"))
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"{clip['index'] + 1:02d}"
    if destination.exists():
        receipt = destination / "render.json"
        if receipt.exists() and read(receipt).get("input_key") == fingerprint(payload):
            value = read(receipt)
            files = [destination / name for name in value["output_hashes"]]
            if all(digest(path) == value["output_hashes"][path.name] for path in files):
                return value, files + [receipt]
        raise CutError("clip output already exists with different or incomplete content")
    with tempfile.TemporaryDirectory(prefix=".clip-", dir=parent) as temporary:
        directory = Path(temporary)
        width, height = info["width"], info["height"]
        caption_report = captions.create(
            shifted, directory, output_seconds(timeline), clip["headline"], config, width, height
        )
        video_report = media.render(
            source, directory, timeline, config, root / "logs" / f"render-{clip['index']:04d}.log"
        )
        report = {
            "input_key": fingerprint(payload),
            "headline": clip["headline"],
            "source_start": clip["start"],
            "source_end": clip["end"],
            "timeline": timeline,
            "captions": caption_report,
            "video": video_report,
            "human_reviewed": False,
            "output_hashes": {
                name: digest(directory / name)
                for name in ("video.mp4", "captions.srt", "captions.ass", "captions.json")
            },
        }
        write(directory / "render.json", report)
        # TemporaryDirectory must not remove the renamed final directory.
        os.rename(directory, destination)
    files = [
        destination / name
        for name in ("video.mp4", "captions.srt", "captions.ass", "captions.json", "render.json")
    ]
    return report, files


def work(project, role, rank, timeout=14400):
    root = Path(project).expanduser().resolve()
    deadline = time.monotonic() + timeout
    models = None
    completed = set()
    while time.monotonic() < deadline:
        try:
            job = read(root / "project.json")
        except (OSError, ValueError):
            time.sleep(2)
            continue
        state = job["state"]
        if state in {"complete", "failed"}:
            if models:
                models.close()
            if state == "failed":
                raise CutError("coordinator marked the project failed; inspect its private logs")
            return state
        config = Config.load(job["config"])
        count = config.gpu_workers if role == "gpu" else config.render_workers
        if not 0 <= rank < count:
            raise CutError("worker rank is outside the configured worker count")
        if role == "gpu" and state in {"rendering", "finalizing"}:
            if models:
                models.close()
            return "gpu_complete"
        stages = ("asr", "align") if role == "gpu" else ("render",)
        for stage in stages:
            active = {"asr": "transcribing", "align": "aligning", "render": "rendering"}[stage]
            if state != active:
                continue
            for path in sorted((root / "tasks" / stage).glob("*.json")):
                spec = read(path)
                if spec["index"] % count != rank or spec["key"] in completed:
                    continue
                if role == "gpu" and models is None:
                    models = Models(config)
                print(f"{role} {rank}: {stage} {spec['index']}", flush=True)
                execute(root, spec, config, models)
                completed.add(spec["key"])
        time.sleep(2)
    raise CutError("worker idle timeout; project was not marked complete")
