import json
import shutil
import subprocess
import wave

import numpy as np
import pytest

from streamcut.captions import create
from streamcut.media import extract_audio, partition, probe, samples, silence_regions
from streamcut.model import Config
from streamcut.pipeline import reexport, run
from streamcut.store import digest, read
from streamcut.timeline import frame_plan, map_words, output_seconds

pytestmark = pytest.mark.integration


@pytest.fixture
def source(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg not installed")
    audio = tmp_path / "source.wav"
    t = np.arange(8 * 48000) / 48000
    value = 0.15 * np.sin(2 * np.pi * 440 * t) * ((t < 2) | (t >= 5))
    with wave.open(str(audio), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(48000)
        f.writeframes((value * 32767).astype("<i2").tobytes())
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=640x360:r=25:d=8",
            "-i",
            str(audio),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        check=True,
    )
    return video


def transcript(tmp_path):
    units = [
        {
            "start": 0,
            "end": 2,
            "text": "最初の話題です",
            "words": [{"start": 0.3, "end": 1.7, "text": "最初の話題です"}],
        },
        {
            "start": 5,
            "end": 8,
            "text": "次の話題です",
            "words": [{"start": 5.2, "end": 7.5, "text": "次の話題です"}],
        },
    ]
    path = tmp_path / "transcript.json"
    path.write_text(json.dumps({"units": units}, ensure_ascii=False))
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {"clips": [{"start_id": "u000001", "end_id": "u000002", "headline": "二つの話題"}]},
            ensure_ascii=False,
        )
    )
    return path, selection


def test_real_ffmpeg_cut_caption_and_resume(source, tmp_path):
    original = digest(source)
    imported, selection = transcript(tmp_path)
    config = Config(
        alignment="provided",
        clips=1,
        min_seconds=1,
        max_seconds=20,
        encoder="libx264",
        silence_cut=True,
    )
    project = tmp_path / "job"
    job = run(str(source), project, config, transcript=imported, selection=selection)
    assert job["state"] == "complete"
    report = read(project / "clips/01/render.json")
    plan = report["timeline"]
    assert plan["frames"] < 200
    assert plan["frames"] > 100
    assert abs(probe(project / "clips/01/video.mp4")["duration"] - output_seconds(plan)) < 0.08
    rows = read(project / "clips/01/captions.json")
    assert "".join(r["text"].replace("\n", "") for r in rows) == "最初の話題です次の話題です"
    assert rows[-1]["start"] < 5.2
    assert all(0 <= r["start"] < r["end"] <= output_seconds(plan) for r in rows)
    first = digest(project / "clips/01/video.mp4")
    assert (
        run(str(source), project, config, resume=True, transcript=imported, selection=selection)[
            "state"
        ]
        == "complete"
    )
    assert digest(project / "clips/01/video.mp4") == first
    assert digest(source) == original
    count = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(project / "clips/01/video.mp4"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert int(count.stdout.strip()) == plan["frames"]

    # A stale complete flag cannot hide missing or changed media.
    (project / "clips/01/video.mp4").write_bytes(b"modified")
    from streamcut.model import CutError

    with pytest.raises(CutError, match="modified"):
        run(str(source), project, config, resume=True, transcript=imported, selection=selection)


def test_shards_reconstruct_pcm_without_gaps_or_duplicates(source, tmp_path):
    root = tmp_path / "parts"
    (root / "audio").mkdir(parents=True)
    audio = root / "audio/master.wav"
    extract_audio(source, audio, root / "extract.log", duration=8)
    shards, _ = partition(audio, 4, root)
    reconstructed = np.concatenate([samples(root / s["audio"]) for s in shards])
    assert np.array_equal(reconstructed, samples(audio))
    assert shards[0]["start"] == 0
    assert all(a["end"] == b["start"] for a, b in zip(shards, shards[1:], strict=False))


def test_loud_wordless_gameplay_is_not_deleted():
    config = Config()
    t = np.arange(4 * 16000) / 16000
    loud = (0.4 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    assert silence_regions(loud, [], config) == []


def test_subtitle_files_share_identical_centisecond_boundaries(tmp_path):
    config = Config(font_size=30)
    plan = frame_plan(0, 3, 3, "30000/1001")
    words = map_words([{"start": 0.1234, "end": 1.876, "text": "字幕の同期確認"}], plan)
    create(words, tmp_path, output_seconds(plan), "確認", config, 640, 360)
    rows = read(tmp_path / "captions.json")
    from streamcut.captions import timestamp

    ass = (tmp_path / "captions.ass").read_text()
    srt = (tmp_path / "captions.srt").read_text()
    for row in rows:
        assert timestamp(row["start"], True) in ass
        assert timestamp(row["start"]) in srt


def test_export_reuses_saved_alignment_without_asr_or_llm(source, tmp_path, monkeypatch):
    imported, selection = transcript(tmp_path)
    config = Config(
        alignment="provided",
        clips=1,
        min_seconds=1,
        max_seconds=20,
        encoder="libx264",
        silence_cut=False,
    )
    project = tmp_path / "export-job"
    run(str(source), project, config, transcript=imported, selection=selection)
    original = digest(project / "clips/01/video.mp4")

    def forbidden(*args, **kwargs):
        raise AssertionError("export must not call ASR, alignment or the LLM")

    monkeypatch.setattr("streamcut.inference.Models.transcribe", forbidden)
    monkeypatch.setattr("streamcut.inference.Models.align", forbidden)
    monkeypatch.setattr("streamcut.planner.plan", forbidden)
    exported = reexport(project, {"font_size": 44}, {"01": "見出しを変更"})
    assert (exported / "clips/01/video.mp4").is_file()
    assert read(exported / "clips/01/render.json")["headline"] == "見出しを変更"
    assert digest(project / "clips/01/video.mp4") == original


def test_non_keyframe_start_matches_requested_source_frame(source, tmp_path):
    from streamcut.media import render

    directory = tmp_path / "non-keyframe"
    directory.mkdir()
    info = probe(source)
    plan = frame_plan(2.04, 3.04, info["duration"], info["fps"])
    render(source, directory, plan, Config(encoder="libx264", burn_in=False), tmp_path / "cut.log")
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(directory / "video.mp4"),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    first = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    differences = []
    for frame in (50, 51, 52):
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(source),
                "-vf",
                f"select=eq(n\\,{frame})",
                "-frames:v",
                "1",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        expected = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        differences.append(np.mean((first - expected) ** 2))
    assert int(np.argmin(differences)) == 1  # source frame 51 = 2.04s at 25 fps


def test_audio_start_offset_is_preserved_in_zero_based_wav(tmp_path):
    video = tmp_path / "delayed.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=s=320x240:r=25:d=2",
            "-itsoffset",
            "0.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1.6",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-t",
            "2",
            str(video),
        ],
        check=True,
    )
    audio = tmp_path / "offset.wav"
    extract_audio(video, audio, tmp_path / "offset.log", duration=2)
    wave_data = samples(audio)
    hop = 320
    rms = np.sqrt(np.mean(wave_data[: len(wave_data) // hop * hop].reshape(-1, hop) ** 2, axis=1))
    onset = np.flatnonzero(rms > 0.02)[0] * 0.02
    assert abs(onset - 0.4) <= 0.04
