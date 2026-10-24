import copy
import json
from fractions import Fraction

import pytest

from streamcut.captions import escape, timestamp
from streamcut.inference import alignment_words
from streamcut.model import Config, CutError, validate_picks, validate_units
from streamcut.planner import chunks
from streamcut.store import cached, read, save_result, task, within
from streamcut.timeline import frame_plan, map_words, output_seconds


def test_rational_frame_plan_does_not_accumulate_drift():
    fps = Fraction(30000, 1001)
    removed = [(i + 0.25, i + 0.7) for i in range(1, 100)]
    plan = frame_plan(0, 101, 101, fps, removed)
    assert plan["frames"] == sum(b - a for a, b in plan["keep_frames"])
    expected = Fraction(plan["frames"]) / fps
    assert abs(output_seconds(plan) - float(expected)) < 1e-10
    for a, b in plan["keep_frames"]:
        assert isinstance(a, int) and isinstance(b, int) and a < b


def test_subtitles_use_same_removed_duration_as_video():
    plan = frame_plan(10, 20, 30, "25", [(12, 15)])
    words = [{"start": 11, "end": 11.5, "text": "前"}, {"start": 16, "end": 17, "text": "後"}]
    mapped = map_words(words, plan)
    assert mapped[0]["start"] == 1
    assert mapped[1]["start"] == 3
    assert output_seconds(plan) == 7


def test_never_duplicate_a_word_across_a_removed_gap():
    plan = frame_plan(0, 10, 10, 30, [(4, 6)])
    with pytest.raises(CutError, match="crosses"):
        map_words([{"start": 3, "end": 7, "text": "単語"}], plan)


def test_alignment_keeps_unsupported_numbers_and_marks_estimates():
    segments = [{"start": 0, "end": 2, "text": "2026年", "chars": []}]
    words, estimated = alignment_words(segments, 2)
    assert "".join(w["text"] for w in words) == "2026年"
    assert estimated == 5
    assert all(0 <= w["start"] < w["end"] <= 2 for w in words)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -1])
def test_untrusted_time_cannot_enter_the_pipeline(value):
    with pytest.raises(CutError):
        validate_units([{"start": value, "end": 10, "text": "字幕"}], 20)


def test_selection_ids_and_overlaps_are_validated():
    units = validate_units(
        [{"start": 0, "end": 30, "text": "話題A"}, {"start": 30, "end": 60, "text": "話題B"}], 60
    )
    config = Config(clips=2)
    with pytest.raises(CutError, match="unknown"):
        validate_picks(
            {"clips": [{"start_id": "made-up", "end_id": "u000002", "headline": "見出し"}]},
            units,
            config,
        )
    pick = {"start_id": "u000001", "end_id": "u000002", "headline": "見出し"}
    with pytest.raises(CutError, match="overlap"):
        validate_picks({"clips": [pick, copy.copy(pick)]}, units, config)


def test_task_receipt_detects_modified_media(tmp_path):
    media = tmp_path / "video.mp4"
    media.write_bytes(b"complete media")
    spec = task(tmp_path, "render", 0, {"source": "input/source.mp4"})
    save_result(tmp_path, spec, {"duration": 1}, [media])
    assert cached(tmp_path, spec) == {"duration": 1}
    media.write_bytes(b"changed media")
    with pytest.raises(CutError, match="modified"):
        cached(tmp_path, spec)


def test_result_json_modification_is_detected(tmp_path):
    spec = task(tmp_path, "asr", 0, {})
    save_result(tmp_path, spec, {"units": []})
    from streamcut.store import result_path

    path = result_path(tmp_path, spec)
    value = json.loads(path.read_text())
    value["value"] = {"units": ["tampered"]}
    path.write_text(json.dumps(value))
    with pytest.raises(CutError, match="modified"):
        cached(tmp_path, spec)


def test_paths_and_nonfinite_json_are_rejected(tmp_path):
    with pytest.raises(CutError):
        within(tmp_path, "../../outside")
    path = tmp_path / "invalid.json"
    path.write_text('{"time": NaN}')
    with pytest.raises(CutError):
        read(path)


def test_caption_text_cannot_inject_ass_overrides():
    assert r"{\pos" not in escape(r"{\pos(1,1)}危険")
    assert timestamp(59.9996) == "00:01:00,000"


def test_planning_chunks_do_not_drop_units():
    units = [{"id": str(i), "text": "あ" * 200} for i in range(50)]
    groups = chunks(units, 2000)
    assert {u["id"] for g in groups for u in g} == {u["id"] for u in units}
    assert all(groups[i][-2:] == groups[i + 1][:2] for i in range(len(groups) - 1))


@pytest.mark.parametrize(
    "settings",
    [{"gpu_workers": 0}, {"batch_size": True}, {"silence_db": float("nan")}, {"unknown": 1}],
)
def test_invalid_configuration_is_rejected(settings):
    with pytest.raises(CutError):
        Config.load(settings)
