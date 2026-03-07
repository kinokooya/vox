# Vox

Windows向けAI音声入力ツール。Push-to-Talk方式（キー長押し）で音声をテキストに変換し、アクティブなテキストフィールドに自動挿入する。

音声認識はすべてローカルで実行され、外部サーバーへの通信は一切行わない。

## 特徴

- **完全ローカル処理** -- 音声データもテキストもネットワークに送信しない
- **高精度STT** -- faster-whisper (Whisper large-v3-turbo) による音声認識。日本語 + 英語技術用語の混在入力に対応
- **低遅延** -- 発話終了から約2秒でテキスト挿入完了
- **LLM整形 (オプション)** -- Ollama (Qwen3-8B) によるフィラー除去・文法修正。必要に応じて `llm.enabled: true` で有効化
- **メディア自動一時停止 (オプション)** -- 録音中にメディア再生を自動で一時停止し、パイプライン完了後に再開。`media.enabled: true` で有効化

## 必要環境

| 項目 | 要件 |
|------|------|
| OS | Windows 10 / 11 |
| GPU | NVIDIA GPU (CUDA対応、VRAM 10GB以上推奨) |
| Python | 3.11以上 |
| マイク | 任意の入力デバイス |
| ストレージ | モデルファイル用に約5GB |
| LLMサーバー | [Ollama](https://ollama.com) (LLM整形を有効にする場合のみ) |

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

# 5. 動作確認
pytest tests/ -v

# 6. 起動
python -m vox
```

初回起動時に faster-whisper のモデル (~3GB) が自動ダウンロードされる。

### LLM整形を有効にする場合（オプション）

```powershell
# Ollama インストール + モデル取得
# https://ollama.com からインストーラをダウンロード・実行
ollama pull qwen3:8b
```

`config.yaml` で `llm.enabled: true` に変更する。

## 使い方

```
python -m vox
```

または Windows ショートカットから `start.bat` を実行（実行時の大きさ: 最小化 推奨）。

終了するには `stop.bat` を実行するか、コンソールで `Ctrl+C` を押す。

起動後、`=== Vox ready — press and hold ctrl_r to speak ===` と表示されれば準備完了。

1. **右Ctrlキーを長押し** -- 録音開始
2. **マイクに向かって話す** -- 最大60秒
3. **右Ctrlキーを離す** -- 録音停止 → STT → テキスト挿入が自動実行される

変換されたテキストがクリップボード経由でアクティブなテキストフィールドに貼り付けられる。

カスタム設定ファイルを指定して起動することもできる:

```
python -m vox path/to/custom-config.yaml
```

## 設定

`config.yaml` で主要パラメータを変更できる。

```yaml
stt:
  engine: "faster-whisper"       # STTエンジン
  word_replacements:             # STT出力の単語置換 (カタカナ→英語表記等)
    クロードコード: "Claude Code"
  faster_whisper:
    model: "large-v3-turbo"      # Whisper モデル
    device: "cuda"
    compute_type: "float16"
    language: "ja"

llm:
  enabled: false                 # true でLLM整形を有効化 (デフォルト: 無効)
  model: "qwen3:8b"              # Ollama モデル名
  base_url: "http://localhost:11434/v1"

hotkey:
  trigger_key: "ctrl_r"          # トリガーキー

audio:
  sample_rate: 16000
  channels: 1
  max_duration_sec: 60           # 最大録音時間 (秒)

insertion:
  pre_paste_delay_ms: 50
  restore_clipboard: true        # 挿入後に元のクリップボードを復元

media:
  enabled: false                 # true で録音中のメディア自動一時停止を有効化
  peak_threshold: 0.01           # 再生中と判定するオーディオ出力レベル (0.0–1.0)
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
         LLM整形 (オプション、有効時のみ)
                |
                v
         テキスト挿入 (clipboard + WM_PASTE)
```

### モジュール構成

```
src/vox/
├── __main__.py    エントリポイント
├── app.py         VoxApp (パイプラインオーケストレータ)
├── config.py      Pydantic 設定モデル
├── hotkey.py      HotkeyListener (pynput)
├── inserter.py    TextInserter (clipboard + WM_PASTE)
├── llm.py         LLMFormatter (OpenAI互換API)
├── media.py       MediaController (録音中メディア自動一時停止)
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
