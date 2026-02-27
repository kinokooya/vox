"""Tests for LLM formatter."""

from unittest.mock import MagicMock, patch

import pytest

from vox.config import LLMConfig
from vox.llm import (
    _FILLERS,
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    LLMFormatter,
)


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


# ---------------------------------------------------------------------------
# Semantic validation tests
# ---------------------------------------------------------------------------

def _make_formatter() -> LLMFormatter:
    """Create a formatter instance for unit-testing internal methods."""
    with patch("vox.llm.OpenAI"):
        return LLMFormatter(LLMConfig())


class TestIsValidFormatting:
    """Tests for _is_valid_formatting: valid edits pass, answers are rejected."""

    def setup_method(self) -> None:
        self.fmt = _make_formatter()

    # --- Should PASS (valid formatting) ---

    def test_punctuation_added(self):
        assert self.fmt._is_valid_formatting(
            "今日はいい天気ですね", "今日はいい天気ですね。"
        )

    def test_filler_removed(self):
        assert self.fmt._is_valid_formatting(
            "えーと今日はいい天気ですね", "今日はいい天気ですね。"
        )

    def test_multiple_fillers_removed(self):
        assert self.fmt._is_valid_formatting(
            "えーとあのーまあ今日はいい天気ですね",
            "今日はいい天気ですね。",
        )

    def test_grammar_correction(self):
        assert self.fmt._is_valid_formatting(
            "バグ直して", "バグを直してください。"
        )

    def test_question_formatted(self):
        assert self.fmt._is_valid_formatting(
            "これどう思いますか", "これどう思いますか。"
        )

    def test_repetition_cleaned(self):
        assert self.fmt._is_valid_formatting(
            "あのーDockerコンテナをあのーKubernetesにデプロイしたい",
            "DockerコンテナをKubernetesにデプロイしたい。",
        )

    def test_identical_input_output(self):
        assert self.fmt._is_valid_formatting("テスト", "テスト")

    def test_empty_input(self):
        assert self.fmt._is_valid_formatting("", "anything")

    # --- Should FAIL (LLM echoed input + added content) ---

    def test_echo_plus_reformulation(self):
        """Real-world case: LLM echoed input and appended a reformulation."""
        assert not self.fmt._is_valid_formatting(
            "いい感じに修正されてますね",
            "いい感じに修正されてますね修正が適切に行われているようです。",
        )

    def test_echo_plus_reformulation_with_period(self):
        assert not self.fmt._is_valid_formatting(
            "いい感じに修正されてますね",
            "いい感じに修正されてますね。修正が適切に行われています。",
        )

    # --- Should FAIL (LLM answered instead of formatting) ---

    def test_answer_to_question(self):
        assert not self.fmt._is_valid_formatting(
            "これより精度を向上させるにはどうすればよいですか",
            "学習データの質と量を向上させ、モデルの訓練回数を増やすことが有効です。",
        )

    def test_translation_to_english(self):
        assert not self.fmt._is_valid_formatting(
            "今日はいい天気ですね",
            "The weather is nice today.",
        )

    def test_explanation_added(self):
        assert not self.fmt._is_valid_formatting(
            "エラーが出ています",
            "このエラーはオブジェクトがnullの場合に発生する例外です。"
            "変数の初期化を確認してください。",
        )

    def test_completely_different_output(self):
        assert not self.fmt._is_valid_formatting(
            "ありがとうございます",
            "どういたしまして。何かあればお気軽にどうぞ。",
        )

    def test_code_suggestion(self):
        assert not self.fmt._is_valid_formatting(
            "Pythonでソートするには",
            "sorted()関数またはlist.sort()メソッドを使用します。"
            "例えば、sorted([3,1,2])で[1,2,3]が返ります。",
        )

    def test_answer_to_accuracy_question(self):
        """Real-world case: LLM answered a question about improving accuracy."""
        assert not self.fmt._is_valid_formatting(
            "もっと精度を向上させるにはどうすればよいですか？",
            "もっと精度を向上させるためには、データの量と質を改善し、"
            "モデルを適切に訓練することが重要です。また、フィードバックループを"
            "設けて実際の結果と比較することで、システムの性能を評価し、"
            "必要に応じて調整することも有効です。",
        )

    def test_long_answer(self):
        assert not self.fmt._is_valid_formatting(
            "機械学習とは",
            "機械学習とは、コンピュータがデータからパターンを学習し、"
            "予測や判断を自動的に行う技術のことです。"
            "教師あり学習、教師なし学習、強化学習の3つに大別されます。",
        )


@patch("vox.llm.OpenAI")
def test_format_text_falls_back_when_llm_answers(mock_openai_cls):
    """format_text falls back to raw text when LLM generates an answer."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    raw_input = "これより精度を向上させるにはどうすればよいですか"
    bad_answer = "学習データの質と量を向上させ、モデルの訓練回数を増やすことが有効です。"

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = bad_answer
    mock_client.chat.completions.create.return_value = mock_response

    config = LLMConfig()
    formatter = LLMFormatter(config)
    result = formatter.format_text(raw_input)

    assert result == raw_input.strip()


@patch("vox.llm.OpenAI")
def test_format_text_falls_back_on_accuracy_question(mock_openai_cls):
    """Real-world regression: LLM answered question about improving accuracy."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    raw_input = "もっと精度を向上させるにはどうすればよいですか？"
    bad_answer = (
        "もっと精度を向上させるためには、データの量と質を改善し、"
        "モデルを適切に訓練することが重要です。また、フィードバックループを"
        "設けて実際の結果と比較することで、システムの性能を評価し、"
        "必要に応じて調整することも有効です。"
    )

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = bad_answer
    mock_client.chat.completions.create.return_value = mock_response

    config = LLMConfig()
    formatter = LLMFormatter(config)
    result = formatter.format_text(raw_input)

    assert result == raw_input.strip()


# ---------------------------------------------------------------------------
# Unit tests for helper methods
# ---------------------------------------------------------------------------


class TestNovelContentRatio:
    def test_identical_text(self):
        ratio = LLMFormatter._novel_content_ratio("テスト入力", "テスト入力")
        assert ratio == 0.0

    def test_completely_novel(self):
        ratio = LLMFormatter._novel_content_ratio("あいう", "かきく")
        assert ratio == 1.0

    def test_partial_overlap(self):
        ratio = LLMFormatter._novel_content_ratio("テスト", "テスト確認")
        assert 0.0 < ratio < 1.0

    def test_empty_output(self):
        ratio = LLMFormatter._novel_content_ratio("テスト", "。！？")
        assert ratio == 0.0


def test_extract_content_chars():
    chars = LLMFormatter._extract_content_chars("テスト test 123。！")
    # Katakana + Latin letters, no digits or punctuation
    assert "テ" in chars
    assert "ス" in chars
    assert "ト" in chars
    assert "t" in chars
    assert "e" in chars
    assert "s" in chars
    assert "1" not in chars
    assert "。" not in chars
    assert "！" not in chars


def test_strip_known_fillers():
    fmt = _make_formatter()
    result = fmt._strip_known_fillers("えーと今日はあのーいい天気ですね")
    assert "えーと" not in result
    assert "あのー" not in result
    assert "今日は" in result
    assert "いい天気ですね" in result


def test_strip_known_fillers_no_fillers():
    fmt = _make_formatter()
    text = "今日はいい天気ですね"
    assert fmt._strip_known_fillers(text) == text


def test_fillers_constant():
    """Ensure _FILLERS contains expected common fillers."""
    assert "えーと" in _FILLERS
    assert "あのー" in _FILLERS
    assert "まあ" in _FILLERS
    assert len(_FILLERS) >= 5
