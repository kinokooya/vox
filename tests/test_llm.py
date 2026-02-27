"""Tests for LLM formatter."""

from unittest.mock import MagicMock, patch

import pytest

from vox.config import LLMConfig
from vox.llm import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, LLMFormatter


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
    # 1 system + 3*2 few-shot + 1 user = 8
    assert len(call_kwargs["messages"]) == 8
    assert call_kwargs["messages"][0]["content"] == SYSTEM_PROMPT
    assert "【音声入力】raw text input【/音声入力】" in call_kwargs["messages"][-1]["content"]


def test_system_prompt_contains_rules():
    assert "フィラー" in SYSTEM_PROMPT
    assert "句読点" in SYSTEM_PROMPT
    assert "質問" in SYSTEM_PROMPT
    assert "英語に変えない" in SYSTEM_PROMPT
    assert "整形ツール" in SYSTEM_PROMPT


def test_llm_config_timeout_sec_default():
    config = LLMConfig()
    assert config.timeout_sec == 30.0


def test_llm_config_timeout_sec_custom():
    config = LLMConfig(timeout_sec=60.0)
    assert config.timeout_sec == 60.0


@patch("vox.llm.OpenAI")
def test_warmup_sends_probe_request(mock_openai_cls):
    """warmup() sends a max_tokens=1 probe to preload the model."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    config = LLMConfig()
    formatter = LLMFormatter(config)
    formatter.warmup()

    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["max_tokens"] == 1
    assert call_kwargs["model"] == config.model


@patch("vox.llm.OpenAI")
def test_warmup_swallows_connection_error(mock_openai_cls):
    """warmup() logs a warning but does not raise if Ollama is unreachable."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.side_effect = ConnectionError("refused")

    config = LLMConfig()
    formatter = LLMFormatter(config)
    # Should not raise
    formatter.warmup()


@patch("vox.llm.OpenAI")
def test_format_text_falls_back_when_output_too_long(mock_openai_cls):
    """When LLM output is much longer than input, fall back to raw text."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    raw_input = "どんな修正がされましたか"
    # Simulate LLM generating a long "answer" instead of formatting
    long_answer = (
        "修正内容は以下の通りです。"
        "まず、システムプロンプトを変更しました。"
        "次に、出力長のガードレールを追加しました。"
    ) * 3

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = long_answer
    mock_client.chat.completions.create.return_value = mock_response

    config = LLMConfig()
    formatter = LLMFormatter(config)
    result = formatter.format_text(raw_input)

    assert result == raw_input.strip()


@patch("vox.llm.OpenAI")
def test_format_text_propagates_timeout_error(mock_openai_cls):
    """When the OpenAI client raises an exception, format_text should propagate it."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("Request timed out")

    config = LLMConfig()
    formatter = LLMFormatter(config)

    with pytest.raises(Exception, match="Request timed out"):
        formatter.format_text("some text")


def test_few_shot_examples_structure():
    """FEW_SHOT_EXAMPLES should have valid user/assistant pairs."""
    assert len(FEW_SHOT_EXAMPLES) >= 2
    for example in FEW_SHOT_EXAMPLES:
        assert "user" in example
        assert "assistant" in example
        assert "【音声入力】" in example["user"]
        assert "【/音声入力】" in example["user"]
        # Assistant output should not contain delimiters
        assert "【音声入力】" not in example["assistant"]


@patch("vox.llm.OpenAI")
def test_format_text_message_structure(mock_openai_cls):
    """Messages should be: system, few-shot pairs, then user with delimiters."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "整形済みテキスト"
    mock_client.chat.completions.create.return_value = mock_response

    config = LLMConfig()
    formatter = LLMFormatter(config)
    formatter.format_text("テスト入力")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]

    # First message is system
    assert messages[0]["role"] == "system"
    # Few-shot pairs alternate user/assistant
    for i, example in enumerate(FEW_SHOT_EXAMPLES):
        assert messages[1 + i * 2]["role"] == "user"
        assert messages[1 + i * 2]["content"] == example["user"]
        assert messages[2 + i * 2]["role"] == "assistant"
        assert messages[2 + i * 2]["content"] == example["assistant"]
    # Last message is the actual user input with delimiters
    last = messages[-1]
    assert last["role"] == "user"
    assert last["content"] == "【音声入力】テスト入力【/音声入力】"


@patch("vox.llm.OpenAI")
def test_format_text_strips_delimiters(mock_openai_cls):
    """If the model echoes back delimiters, they should be stripped."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "【音声入力】テスト出力【/音声入力】"
    mock_client.chat.completions.create.return_value = mock_response

    config = LLMConfig()
    formatter = LLMFormatter(config)
    result = formatter.format_text("テスト入力")

    assert result == "テスト出力"
