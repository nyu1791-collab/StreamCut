import io
import json
import urllib.error

import pytest

from streamcut.model import Config, CutError
from streamcut.planner import complete


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
