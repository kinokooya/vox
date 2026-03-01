# Vox - AI音声入力ツール

## 最初に読むこと

**開発の全経緯・アーキテクチャ・次のタスクは `docs/handoff.md` に詳細記載。必ず最初に読むこと。**

## プロジェクト概要

Windows向けデスクトップ常駐型の音声入力ツール。Push-to-Talk方式（右Ctrl長押し）でマイク入力を受け付け、ローカルSTT (faster-whisper) でテキスト変換し、アクティブなテキストフィールドにクリップボード経由で自動挿入する。オプションでローカルLLM (Ollama/Qwen3) によるテキスト整形も可能（`llm.enabled` で切替）。

- 対象環境: Windows 10/11, RTX 5070 Ti (16GB VRAM)
- 言語: Python 3.11+
- 詳細仕様: `requirements-specification.md`
- 開発ログ・引き継ぎ: `docs/handoff.md`

## 現在のステータス

- **Phase 1 MVP: 実装完了・Windows 実機テスト済み** (テスト114件全通過)
- Phase 2: 未着手

## ファイル構造

```
src/vox/
├── __main__.py          # エントリポイント
├── app.py               # VoxApp パイプラインオーケストレータ
├── config.py            # Pydantic設定 (AppConfig)
├── hotkey.py            # HotkeyListener (pynput, 右Ctrl)
├── inserter.py          # TextInserter (clipboard + WM_PASTE)
├── llm.py               # LLMFormatter (OpenAI互換API)
├── media.py             # MediaController (録音中メディア自動一時停止)
├── recorder.py          # AudioRecorder (sounddevice)
└── stt/
    ├── base.py           # STTEngine ABC
    ├── factory.py        # create_stt_engine()
    └── faster_whisper_engine.py

start.bat                # Windows ショートカット用起動スクリプト
start.sh                 # bash 用起動スクリプト
stop.bat                 # プロセス終了スクリプト (PIDファイル経由)
```

## 開発ルール

### タスク管理

- **作業内容・タスク・進捗はすべて `docs/handoff.md` に記述する**
- GitHub Issues は外部からの報告用のみ。自分のタスク管理には使わない

### Git ワークフロー

- **feature ブランチで開発し、main へ PR を出してマージする**
- ブランチ名: `feature/<内容>`, `fix/<内容>`, `refactor/<内容>`
- main への直接 push は原則禁止
- PR 作成前に lint + テストを通すこと
- CI (GitHub Actions) が通ることを確認してからマージする

### Git コミット

- **適時コミットすること** — 意味のある単位（機能追加、バグ修正、リファクタリング等）で都度コミットする
- コミットメッセージは英語で、変更の「why」を簡潔に記述する
- 大きな変更は小さなコミットに分割する
- コミット前に lint + テストを通すこと

### コードレビュー / リファクタリング

- 実装がある程度まとまったら Codex にコードレビュー・リファクタリングを依頼する
- レビュー観点: コード品質、パフォーマンス、スレッド安全性、可読性

### GitHub Actions

- CI は `.github/workflows/ci.yml` に定義
- PR 作成時・push 時に自動で ruff lint + mypy + pytest を実行

## コマンド

```bash
# 依存インストール
pip install -e ".[dev]"

# lint
ruff check src/ tests/

# type check
mypy src/

# test
pytest tests/ -v

# 実行
python -m vox
# または: python -m vox path/to/custom-config.yaml

# Windows ショートカットから起動
# start.bat をショートカットに登録（実行時の大きさ: 最小化）
```

## アーキテクチャ要点

- **パイプライン**: 右Ctrl押下→録音→メディア一時停止(opt-in)→右Ctrl解放→STT→(LLM)→クリップボード挿入→メディア再開
- **STT抽象化**: `STTEngine` ABC。Phase 1 は faster-whisper、Phase 2 で SenseVoice 追加予定
- **LLM**: OpenAI互換API (Ollama `http://localhost:11434/v1`)。`config.yaml` でモデル変更可。`llm.enabled: false` で無効化（STT出力をそのまま挿入）
- **スレッド**: パイプラインは別スレッドで実行。排他制御で二重実行防止。shutdown時join
- **設定**: `config.yaml` → Pydantic モデル。型安全 + バリデーション付き
