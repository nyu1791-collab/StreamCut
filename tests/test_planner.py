import io
import json
import urllib.error

import pytest

from streamcut.model import Config, CutError, validate_units
from streamcut.planner import complete, plan


def test_confirmed_llm_response_is_cached_without_resending(tmp_path, monkeypatch):
    monkeypatch.setenv("STREAMCUT_API_KEY", "unit-test-only")
    calls = []

    def response(request, **kwargs):
        calls.append(request)
        return io.BytesIO(
            json.dumps(
                {"choices": [{"finish_reason": "stop", "message": {"content": '{"clips": []}'}}]}
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", response)
    config = Config(llm_model="explicit-test-model")
    assert complete(tmp_path, "test prompt", config) == {"clips": []}
    assert complete(tmp_path, "test prompt", config) == {"clips": []}
    assert len(calls) == 1
    saved = next((tmp_path / "cache/planning").glob("*.json"))
    assert "unit-test-only" not in saved.read_text()


def test_provider_failure_is_not_automatically_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("STREAMCUT_API_KEY", "unit-test-only")
    calls = []

    def response(request, **kwargs):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 429, "limited", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", response)
    config = Config(llm_model="explicit-test-model")
    with pytest.raises(CutError, match="HTTP 429"):
        complete(tmp_path, "test prompt", config)
    with pytest.raises(CutError, match="confirmed response"):
        complete(tmp_path, "test prompt", config)
    assert len(calls) == 1


def test_truncated_response_cannot_be_used_as_a_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("STREAMCUT_API_KEY", "unit-test-only")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: io.BytesIO(
            json.dumps(
                {"choices": [{"finish_reason": "length", "message": {"content": '{"clips": []}'}}]}
            ).encode()
        ),
    )
    with pytest.raises(CutError, match="truncated"):
        complete(tmp_path, "prompt", Config(llm_model="explicit-test-model"))


def test_single_chunk_combines_topics_and_clip_selection_in_one_api_call(tmp_path, monkeypatch):
    calls = []
    chosen = {"start_id": "u000001", "end_id": "u000002", "headline": "話題"}

    def response(root, prompt, config, retry):
        calls.append(prompt)
        return {
            "topics": [
                {
                    "start_id": "u000001",
                    "end_id": "u000002",
                    "title": "話題",
                    "summary": "前振りと結末",
                }
            ],
            "clips": [chosen],
        }

    monkeypatch.setattr("streamcut.planner.complete", response)
    units = validate_units(
        [{"start": 0, "end": 30, "text": "前振り"}, {"start": 30, "end": 60, "text": "結末"}], 60
    )
    assert plan(tmp_path, units, Config(llm_model="test")) == {"clips": [chosen]}
    assert len(calls) == 1
    assert (tmp_path / "topics.json").is_file()
