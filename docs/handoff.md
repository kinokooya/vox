# Vox 開発引き継ぎドキュメント

> このファイルは WSL2 上の開発セッションから Windows 上の次セッションへの引き継ぎ用です。
> **最終更新**: 2025-02-19 (WSL2 上で Phase 1 MVP 完了)

---

## 1. これまでの作業サマリー

### 完了済み

| # | 作業内容 | コミット |
|---|---------|---------|
| 1 | 要件定義書 v1.1 作成 (STT複数対応, LLMサイズ柔軟化, Deep Context追加) | `e0fb6b0` |
| 2 | CLAUDE.md + GitHub Actions CI (.github/workflows/ci.yml) | `ef2f492` |
| 3 | Phase 1 MVP 全モジュール実装 | `be6609a` |
| 4 | Codex によるコードレビュー → 指摘修正 (HIGH×3, MEDIUM×4, LOW×3) | `1e2cfe1` |

### 未実施

- **Windows 実機テスト** — WSL2 ではマイク/クリップボード/Win32 API が動かないため未検証
- **Ollama との結合テスト** — Ollama サーバーが未起動のため未検証
- **faster-whisper モデルロード** — GPU 実機が必要
- **Phase 2 以降の実装**

---

## 2. プロジェクト構造

```
vox/
├── .github/workflows/ci.yml        # GitHub Actions: ruff + mypy + pytest
├── CLAUDE.md                        # Claude Code 用プロジェクトガイド
├── config.yaml                      # デフォルト設定ファイル
├── pyproject.toml                   # 依存関係 + ツール設定
├── requirements-specification.md    # 要件定義書 v1.1
├── docs/
│   └── handoff.md                   # ← このファイル
├── src/vox/
│   ├── __init__.py                  # パッケージ定義 (version=0.1.0)
│   ├── __main__.py                  # エントリポイント: python -m vox
│   ├── app.py                       # VoxApp: パイプライン統合オーケストレータ
│   ├── config.py                    # Pydantic設定モデル (AppConfig等9クラス)
│   ├── hotkey.py                    # HotkeyListener: pynput右Alt Hold検出
│   ├── inserter.py                  # TextInserter: clipboard + Win32 Ctrl+V
│   ├── llm.py                       # LLMFormatter: OpenAI互換API呼び出し
│   ├── recorder.py                  # AudioRecorder: sounddevice録音
│   └── stt/
│       ├── __init__.py              # 公開API: STTEngine, create_stt_engine
│       ├── base.py                  # STTEngine ABC (抽象基底クラス)
│       ├── factory.py               # エンジンファクトリ (config → 実装選択)
│       └── faster_whisper_engine.py # FasterWhisperEngine 実装
└── tests/
    ├── test_config.py               # 設定テスト (6件)
    ├── test_llm.py                  # LLMフォーマッタテスト (3件)
    └── test_stt.py                  # STTインターフェーステスト (4件)
```

---

## 3. アーキテクチャ

### 処理パイプライン

```
右Alt押下 → AudioRecorder.start()
右Alt解放 → AudioRecorder.stop() → STTEngine.transcribe() → LLMFormatter.format_text() → TextInserter.insert()
```

- `VoxApp` (app.py) がオーケストレータ
- `HotkeyListener` がキーイベントを検出し、VoxApp のコールバックを呼ぶ
- パイプラインは別スレッドで実行（ホットキーリスナーをブロックしない）
- 処理中は排他制御（`_processing` フラグ）で二重実行を防止

### STT 抽象化

```
STTEngine (ABC)
├── FasterWhisperEngine  ← Phase 1 で実装済み
└── SenseVoiceEngine     ← Phase 2 で実装予定 (factory.py に NotImplementedError)
```

### 設定

`config.yaml` → `AppConfig` (Pydantic) で読み込み。各モジュールは自分の Config を受け取る。

---

## 4. Windows 環境構築手順

### 前提
- Windows 10/11
- RTX 5070 Ti + 最新 NVIDIA ドライバ
- マイク接続済み

### セットアップ

```powershell
# 1. Python 3.11+ を確認（なければ https://python.org からインストール）
python --version

# 2. プロジェクトを取得
# WSL から: \\wsl$\Ubuntu\home\above0821\projects\vox を C:\Users\<user>\projects\vox にコピー
# または Git リモートがあれば: git clone <url>

# 3. venv + 依存インストール
cd C:\Users\<user>\projects\vox
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# 4. Ollama インストール + モデル取得
# https://ollama.com からインストーラをダウンロード・実行
ollama pull qwen2.5:7b-instruct-q4_K_M
# Ollama はインストール後自動的にバックグラウンドで起動する

# 5. テスト実行（環境確認）
pytest tests/ -v
ruff check src/

# 6. アプリ起動
python -m vox
# → 「=== Vox ready — press and hold alt_r to speak ===」が出れば成功
# → 右Altを押しながら話し、離すとパイプラインが実行される
```

### トラブルシューティング

| 症状 | 原因・対処 |
|------|-----------|
| `CUDA error` / GPU 認識しない | NVIDIA ドライバ更新。`nvidia-smi` で確認 |
| `ctranslate2` インポートエラー | `pip install ctranslate2 --upgrade --force-reinstall` |
| faster-whisper モデルダウンロードが遅い | 初回のみ ~3GB DL。HuggingFace から。VPN があれば切る |
| Ollama 接続エラー | `ollama serve` が起動しているか確認。`curl http://localhost:11434/v1/models` |
| マイクが認識されない | Windows のサウンド設定でデフォルト入力デバイスを確認 |
| 右Alt が反応しない | 一部キーボードは Alt_R を AltGr として扱う。config.yaml の `trigger_key` を変更 |

---

## 5. 次にやるべきこと（優先順）

### 即座に必要

1. **Windows 実機で `python -m vox` を起動して動作確認**
2. 動かない場合はエラーを見て修正
3. 動いたら実際にマイクで話して E2E テスト

### Phase 2 タスク（要件定義書 Section 7 参照）

| # | タスク | 概要 |
|---|--------|------|
| 6 | ポップアップ/オーバーレイ | tkinter で画面右下に状態表示 |
| 7 | システムトレイ | pystray でトレイアイコン |
| 8 | タイムアウト処理 | STT 30s / LLM 30s。LLM タイムアウト時は生テキストフォールバック |
| 9 | VRAM フォールバック | CUDA エラー時のモデル再ロード |
| 10 | SenseVoice 対応 | SenseVoiceEngine 実装 + funasr 依存追加 |
| 11 | Deep Context | Windows UI Automation でアクティブウィンドウのテキスト取得 → LLM コンテキスト注入 |

---

## 6. コードレビューで修正済みの項目

Codex による自動レビューで以下を修正済み：

- **[HIGH]** AudioRecorder: stream 作成失敗時のロールバック、stop() の冪等性
- **[HIGH]** HotkeyListener: コールバック例外の catch（リスナースレッド保護）
- **[HIGH]** VoxApp: shutdown 時にワーカースレッドを join（処理中の安全な終了）
- **[MEDIUM]** TextInserter: クリップボード復元の安全化
- **[MEDIUM]** FasterWhisperEngine: sample_rate 16kHz バリデーション
- **[MEDIUM]** config: YAML の UTF-8 エンコーディング指定
- **[LOW]** AudioRecorder: O(n²) → O(1) のフレームカウント
- **[LOW]** Config: Pydantic Field で数値バウンド制約追加

---

## 7. 重要な設計判断

| 判断 | 理由 |
|------|------|
| OpenAI SDK で LLM 呼び出し | Ollama / LM Studio 両方が OpenAI 互換 API を提供するため |
| STTEngine ABC パターン | Phase 2 で SenseVoice 追加時に差し替え可能にするため |
| パイプラインを別スレッド実行 | ホットキーリスナーをブロックしないため |
| Win32 keybd_event で Ctrl+V | pyautogui より軽量。フォーカスを奪わない |
| Pydantic + YAML | 型安全 + 人間が読み書きしやすい設定ファイル |
