"""Tests for LLM formatter."""

from unittest.mock import MagicMock, patch

from vox.config import LLMConfig
from vox.llm import SYSTEM_PROMPT, LLMFormatter


def test_format_empty_text():
    config = LLMConfig()
    formatter = LLMFormatter(config)
    assert formatter.format_text("") == ""
    assert formatter.format_text("   ") == ""


@patch("vox.llm.OpenAI")
def test_format_text_calls_api(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "formatted text"
    mock_client.chat.completions.create.return_value = mock_response

    config = LLMConfig()
    formatter = LLMFormatter(config)
    result = formatter.format_text("raw text input")

    assert result == "formatted text"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == config.model
    assert call_kwargs["temperature"] == config.temperature
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["content"] == SYSTEM_PROMPT
    assert call_kwargs["messages"][1]["content"] == "raw text input"


def test_system_prompt_contains_rules():
    assert "フィラー" in SYSTEM_PROMPT
    assert "句読点" in SYSTEM_PROMPT
    assert "技術用語" in SYSTEM_PROMPT
