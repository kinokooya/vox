# Vox

Windows向けAI音声入力ツール。Push-to-Talk方式（キー長押し）で音声をテキストに変換し、アクティブなテキストフィールドに自動挿入する。

音声認識とテキスト整形はすべてローカルで実行され、外部サーバーへの通信は一切行わない。

## 特徴

- **完全ローカル処理** -- 音声データもテキストもネットワークに送信しない
- **ローカルSTT** -- faster-whisper (Whisper large-v3-turbo) による高精度な音声認識
- **ローカルLLM整形** -- Ollama (Qwen2.5-7B) でフィラー除去・文法修正・文章再構成を自動実行
- **低遅延** -- 15秒程度の発話に対して約2〜5秒でテキスト挿入完了
- **日本語 + 英語技術用語の混在入力に対応**

## 必要環境

| 項目 | 要件 |
|------|------|
| OS | Windows 10 / 11 |
| GPU | NVIDIA GPU (CUDA対応、VRAM 10GB以上推奨) |
| Python | 3.11以上 |
| LLMサーバー | [Ollama](https://ollama.com) |
| マイク | 任意の入力デバイス |
| ストレージ | モデルファイル用に約15GB |

## セットアップ

```powershell
# 1. Python インストール
winget install Python.Python.3.12

# 2. リポジトリ取得
git clone <url>
cd vox

# 3. venv 作成 + 依存インストール
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# 4. CUDA ランタイム (pip 経由、PATH 設定不要)
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12

# 5. Ollama インストール + モデル取得
# https://ollama.com からインストーラをダウンロード・実行
ollama pull qwen2.5:7b-instruct-q4_K_M

# 6. 動作確認
pytest tests/ -v

# 7. 起動
python -m vox
```

初回起動時に faster-whisper のモデル (~3GB) が自動ダウンロードされる。

## 使い方

```
python -m vox
```

起動後、`=== Vox ready — press and hold ctrl_r to speak ===` と表示されれば準備完了。

1. **右Ctrlキーを長押し** -- 録音開始
2. **マイクに向かって話す** -- 最大60秒
3. **右Ctrlキーを離す** -- 録音停止、STT → LLM整形 → テキスト挿入が自動実行される

整形済みテキストがクリップボード経由でアクティブなテキストフィールドに貼り付けられる。

カスタム設定ファイルを指定して起動することもできる:

```
python -m vox path/to/custom-config.yaml
```

## 設定

`config.yaml` で主要パラメータを変更できる。

```yaml
stt:
  engine: "faster-whisper"       # STTエンジン ("faster-whisper" or "sensevoice")
  faster_whisper:
    model: "large-v3-turbo"      # Whisper モデル
    device: "cuda"
    compute_type: "float16"
    language: "ja"

llm:
  model: "qwen2.5:7b-instruct-q4_K_M"  # Ollama モデル名
  base_url: "http://localhost:11434/v1"
  temperature: 0.3
  max_tokens: 1024

hotkey:
  trigger_key: "ctrl_r"          # トリガーキー

audio:
  sample_rate: 16000
  channels: 1
  max_duration_sec: 60           # 最大録音時間 (秒)

insertion:
  pre_paste_delay_ms: 50
  restore_clipboard: true        # 挿入後に元のクリップボードを復元
```

## アーキテクチャ

### 処理パイプライン

```
右Ctrl押下 --> 録音開始 (sounddevice)
右Ctrl解放 --> 録音停止
                |
                v
         STT (faster-whisper)
                |
                v
         LLM整形 (Ollama / Qwen2.5)
                |
                v
         テキスト挿入 (clipboard + Ctrl+V)
```

### モジュール構成

```
src/vox/
├── __main__.py    エントリポイント
├── app.py         VoxApp (パイプラインオーケストレータ)
├── config.py      Pydantic 設定モデル
├── hotkey.py      HotkeyListener (pynput)
├── inserter.py    TextInserter (clipboard + Win32 Ctrl+V)
├── llm.py         LLMFormatter (OpenAI互換API)
├── recorder.py    AudioRecorder (sounddevice)
└── stt/
    ├── base.py              STTEngine ABC
    ├── factory.py           エンジンファクトリ
    └── faster_whisper_engine.py
```

- `VoxApp` が全モジュールを統合し、パイプラインを制御する
- パイプラインは別スレッドで実行され、ホットキーリスナーをブロックしない
- 排他制御により処理中の二重実行を防止する
- STTエンジンは `STTEngine` ABC で抽象化されており、実装の差し替えが可能

## 開発

```bash
# 依存インストール
pip install -e ".[dev]"

# Lint
ruff check src/ tests/

# 型チェック
mypy src/

# テスト
pytest tests/ -v
```

GitHub Actions CI (`.github/workflows/ci.yml`) により、PR作成時・push時に ruff + mypy + pytest が自動実行される。

## ライセンス

MIT
