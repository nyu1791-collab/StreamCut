from concurrent.futures import ThreadPoolExecutor

import pytest
from test_media import source as media_source

from streamcut.inference import Models
from streamcut.model import Config, CutError
from streamcut.pipeline import run
from streamcut.store import read, write
from streamcut.worker import work

source = media_source


@pytest.mark.integration
def test_two_gpu_workers_and_two_cpu_workers_complete_shared_project(source, tmp_path, monkeypatch):  # noqa: F811
    calls = {"asr": [], "align": []}

    def asr(self, root, payload):
        calls["asr"].append(payload["index"])
        return {
            "units": [
                {
                    "start": payload["start"],
                    "end": payload["end"],
                    "text": "並列処理の確認です",
                    "words": [],
                }
            ],
            "language": "ja",
        }

    def align(self, root, payload):
        calls["align"].append(payload["origin"])
        unit = payload["units"][0]
        return {
            "words": [
                {"start": unit["start"] + 0.3, "end": unit["end"] - 0.3, "text": unit["text"]}
            ],
            "estimated_characters": 0,
            "alignment": "mock_for_scheduler_test",
        }

    monkeypatch.setattr(Models, "transcribe", asr)
    monkeypatch.setattr(Models, "align", align)
    selection = tmp_path / "selection.json"
    write(
        selection,
        {
            "clips": [
                {"start_id": "u000001", "end_id": "u000001", "headline": "前半"},
                {"start_id": "u000002", "end_id": "u000002", "headline": "後半"},
            ]
        },
    )
    config = Config(
        gpu_workers=2,
        render_workers=2,
        clips=2,
        min_seconds=1,
        max_seconds=20,
        encoder="libx264",
        silence_cut=False,
        worker_timeout=30,
    )
    project = tmp_path / "distributed"
    with ThreadPoolExecutor(max_workers=3) as pool:
        workers = [pool.submit(work, project, "gpu", i, 30) for i in range(2)]
        workers.append(pool.submit(work, project, "render", 1, 30))
        result = run(str(source), project, config, distributed=True, selection=selection)
        assert all(f.result() in {"complete", "gpu_complete"} for f in workers)
    assert result["state"] == "complete"
    assert sorted(calls["asr"]) == [0, 1]
    assert len(calls["align"]) == 2
    assert all((project / f"clips/{i:02d}/video.mp4").exists() for i in (1, 2))


@pytest.mark.integration
def test_worker_failure_propagates_without_persisting_exception_payload(
    source, tmp_path, monkeypatch
):  # noqa: F811
    def fail(self, root, payload):
        raise RuntimeError("private provider payload must not be in receipts")

    monkeypatch.setattr(Models, "transcribe", fail)
    selection = tmp_path / "selection.json"
    write(selection, {"clips": []})
    project = tmp_path / "failed"
    config = Config(worker_timeout=30)
    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(work, project, "gpu", 0, 30)
        with pytest.raises(CutError, match="worker failed"):
            run(str(source), project, config, distributed=True, selection=selection)
        with pytest.raises(RuntimeError):
            worker.result()
    assert read(project / "project.json")["state"] == "failed"
    receipts = list((project / "results/asr").glob("*.json"))
    assert receipts
    assert "private provider" not in receipts[0].read_text()


@pytest.mark.integration
def test_four_gpu_ranks_and_four_cpu_ranks_render_four_clips(source, tmp_path, monkeypatch):
    calls = []

    def align(self, root, payload):
        unit = payload["units"][0]
        calls.append(unit["id"])
        return {
            "words": [
                {"start": unit["start"] + 0.2, "end": unit["end"] - 0.2, "text": unit["text"]}
            ],
            "estimated_characters": 0,
            "alignment": "mock_for_four_runtime_scheduler_test",
        }

    monkeypatch.setattr(Models, "align", align)
    imported = tmp_path / "imported.json"
    write(
        imported,
        {
            "units": [
                {"start": 2 * i, "end": 2 * i + 2, "text": "四台で処理します"} for i in range(4)
            ]
        },
    )
    selection = tmp_path / "four.json"
    write(
        selection,
        {
            "clips": [
                {"start_id": f"u{i + 1:06d}", "end_id": f"u{i + 1:06d}", "headline": f"話題{i + 1}"}
                for i in range(4)
            ]
        },
    )
    config = Config(
        gpu_workers=4,
        render_workers=4,
        clips=4,
        min_seconds=1,
        max_seconds=20,
        encoder="libx264",
        silence_cut=False,
        worker_timeout=40,
    )
    project = tmp_path / "four-ranks"
    with ThreadPoolExecutor(max_workers=7) as pool:
        workers = [pool.submit(work, project, "gpu", i, 40) for i in range(4)]
        workers += [pool.submit(work, project, "render", i, 40) for i in range(1, 4)]
        result = run(
            str(source), project, config, distributed=True, transcript=imported, selection=selection
        )
        assert all(f.result() in {"complete", "gpu_complete"} for f in workers)
    assert result["state"] == "complete"
    assert sorted(calls) == [f"u{i:06d}" for i in range(1, 5)]
    assert all((project / f"clips/{i:02d}/video.mp4").exists() for i in range(1, 5))
