"""Tests for STT hallucination prevention and Chinese detection."""

import pytest

from vox.config import FasterWhisperConfig
from vox.stt.faster_whisper_engine import FasterWhisperEngine


@pytest.fixture()
def engine() -> FasterWhisperEngine:
    config = FasterWhisperConfig()
    return FasterWhisperEngine(config)


class TestValidateTranscription:
    """Tests for _validate_transcription post-processing."""

    def test_empty_text_returns_empty(self, engine: FasterWhisperEngine):
        assert engine._validate_transcription("", 1.0) == ""

    def test_normal_text_passes(self, engine: FasterWhisperEngine):
        assert engine._validate_transcription("こんにちは", 2.0) == "こんにちは"

    def test_hallucination_pattern_go_shichou(self, engine: FasterWhisperEngine):
        assert engine._validate_transcription("ご視聴ありがとうございました", 1.0) == ""

    def test_hallucination_pattern_channel(self, engine: FasterWhisperEngine):
        assert engine._validate_transcription("チャンネル登録お願いします", 1.0) == ""

    def test_hallucination_pattern_high_rating(self, engine: FasterWhisperEngine):
        assert engine._validate_transcription("高評価お願いします", 1.0) == ""

    def test_repetition_detected(self, engine: FasterWhisperEngine):
        # Same phrase repeated 3+ times
        assert engine._validate_transcription("あいうあいうあいう", 3.0) == ""

    def test_onegai_shimasu_not_false_positive(self, engine: FasterWhisperEngine):
        text = "しっかりと頑張るのでよろしくお願いします。"
        assert engine._validate_transcription(text, 5.0) == text

    def test_no_false_positive_on_short_repeat(self, engine: FasterWhisperEngine):
        # Only 2 repetitions should pass
        assert engine._validate_transcription("はいはい", 1.0) == "はいはい"

    def test_suspicious_char_ratio_short_audio(self, engine: FasterWhisperEngine):
        # 2 second audio producing 50 chars = 25 chars/sec (suspicious)
        long_text = "あ" * 50
        assert engine._validate_transcription(long_text, 2.0) == ""

    def test_normal_char_ratio_passes(self, engine: FasterWhisperEngine):
        # 5 second audio producing 30 chars = 6 chars/sec (normal)
        text = "今日はいい天気ですね。散歩に行きましょう。"
        assert engine._validate_transcription(text, 5.0) == text

    def test_char_ratio_not_checked_for_long_audio(
        self, engine: FasterWhisperEngine
    ):
        # For audio >= 3s, ratio check is skipped
        long_text = (
            "今日はとても良い天気で散歩に行きましょう。"
            "公園で友達と会いました。楽しかったです。"
        )
        # 3.0s audio, many chars — but duration >= 3s so no ratio check
        result = engine._validate_transcription(long_text, 3.0)
        assert result == long_text


class TestSimplifiedChineseDetection:
    """Tests for simplified Chinese character detection."""

    def test_simplified_chinese_detected(self, engine: FasterWhisperEngine):
        # Contains simplified Chinese chars like 这、们
        assert engine._contains_simplified_chinese("这是一个测试") is True

    def test_japanese_text_not_flagged(self, engine: FasterWhisperEngine):
        assert engine._contains_simplified_chinese("これはテストです") is False

    def test_japanese_kanji_not_flagged(self, engine: FasterWhisperEngine):
        # Common Japanese kanji (繁体字-based) should not trigger
        assert engine._contains_simplified_chinese("東京都渋谷区") is False

    def test_mixed_triggers_detection(self, engine: FasterWhisperEngine):
        # Japanese + one simplified Chinese char
        assert engine._contains_simplified_chinese("テスト这") is True

    def test_validate_discards_chinese_output(self, engine: FasterWhisperEngine):
        assert engine._validate_transcription("这是什么意思", 2.0) == ""

    def test_validate_keeps_japanese(self, engine: FasterWhisperEngine):
        text = "音声認識のテストです"
        assert engine._validate_transcription(text, 3.0) == text
