# Vox - AI音声入力ツール

## プロジェクト概要

Windows向けデスクトップ常駐型の音声入力ツール。Push-to-Talk方式でマイク入力を受け付け、ローカルSTT + ローカルLLMでテキスト整形し、アクティブなテキストフィールドに自動挿入する。

- 対象環境: Windows 10/11, RTX 5070 Ti (16GB VRAM)
- 言語: Python 3.11+
- 詳細仕様: `requirements-specification.md` を参照

## 開発ルール

### Git コミット

- **適時コミットすること** — 意味のある単位（機能追加、バグ修正、リファクタリング等）で都度コミットする
- コミットメッセージは英語で、変更の「why」を簡潔に記述する
- 大きな変更は小さなコミットに分割する
- コミット前に動作確認すること

### コードレビュー / リファクタリング

- 実装がある程度まとまったら Codex にコードレビュー・リファクタリングを依頼する
- レビュー観点: コード品質、パフォーマンス、セキュリティ、可読性

### GitHub Actions

- CI は `.github/workflows/` に定義
- PR 作成時・push 時に自動で lint + テストを実行

## アーキテクチャ

### STTエンジン抽象化

`STTEngine` ABC で faster-whisper / SenseVoice を切り替え可能にする。
設定は `config.yaml` の `stt.engine` フィールドで指定。

### LLM

Ollama (OpenAI互換API) 経由で Qwen2.5 を利用。
7B / 3B を `config.yaml` で切り替え可能。

## コマンド

```bash
# lint
ruff check src/

# type check
mypy src/

# test
pytest tests/

# run
python -m vox
```
