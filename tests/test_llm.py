"""Tests for LLM formatter."""

from unittest.mock import MagicMock, patch

import pytest

from vox.config import LLMConfig
from vox.llm import SYSTEM_PROMPT, LLMFormatter, LLMPermanentError, LLMTransientError


def test_format_empty_text() -> None:
    config = LLMConfig()
    formatter = LLMFormatter(config)
    assert formatter.format_text("") == ""
    assert formatter.format_text("   ") == ""


@patch("vox.llm.OpenAI")
def test_format_text_calls_api(mock_openai_cls) -> None:
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "formatted text"
    mock_client.chat.completions.create.return_value = mock_response

    config = LLMConfig(timeout_sec=12.5)
    formatter = LLMFormatter(config)
    result = formatter.format_text("raw text input")

    assert result == "formatted text"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == config.model
    assert call_kwargs["temperature"] == config.temperature
    assert call_kwargs["timeout"] == config.timeout_sec
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["content"] == SYSTEM_PROMPT
    assert call_kwargs["messages"][1]["content"] == "raw text input"


def test_retries_transient_error_then_succeeds() -> None:
    calls = {"count": 0}

    class DummyTimeoutError(Exception):
        pass

    def create(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise DummyTimeoutError("temporary timeout")
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "ok"
        return response

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    sleeps: list[float] = []

    formatter = LLMFormatter(
        LLMConfig(retry_count=2, retry_backoff_sec=0.2),
        client=client,
        sleep_fn=sleeps.append,
    )
    assert formatter.format_text("hello") == "ok"
    assert calls["count"] == 2
    assert sleeps == [0.2]


def test_transient_error_exhausts_retries() -> None:
    class DummyTimeoutError(Exception):
        pass

    client = MagicMock()
    client.chat.completions.create.side_effect = DummyTimeoutError("still failing")
    sleeps: list[float] = []

    formatter = LLMFormatter(
        LLMConfig(retry_count=1, retry_backoff_sec=0.1),
        client=client,
        sleep_fn=sleeps.append,
    )

    with pytest.raises(LLMTransientError):
        formatter.format_text("hello")

    assert client.chat.completions.create.call_count == 2
    assert sleeps == [0.1]


def test_non_retryable_error_raises_permanent_error() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = ValueError("bad request")

    formatter = LLMFormatter(LLMConfig(retry_count=3), client=client, sleep_fn=lambda _v: None)

    with pytest.raises(LLMPermanentError):
        formatter.format_text("hello")

    assert client.chat.completions.create.call_count == 1


def test_system_prompt_contains_rules() -> None:
    assert "????" in SYSTEM_PROMPT
    assert "???" in SYSTEM_PROMPT
    assert "????" in SYSTEM_PROMPT
