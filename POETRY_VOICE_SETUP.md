# 🎤 Poetry環境での音声認識セットアップガイド

Poetry環境で音声認識システムを有効にする手順です。

## 🚀 クイックセットアップ

### **1. システム依存関係のインストール（macOS）**
```bash
# portaudioをHomebrew経由でインストール
brew install portaudio
```

### **2. Poetry経由での音声認識ライブラリインストール**

#### **オプション A: 音声認識のみ**
```bash
poetry install --extras voice
```

#### **オプション B: 全機能（推奨）**
```bash
poetry install --extras full
```

#### **オプション C: 個別インストール**
```bash
poetry add speechrecognition pyaudio
```

### **3. 環境アクティベート & 実行**
```bash
poetry shell
python -m app.examples.web_camera_oo
```

## 📋 詳細インストールオプション

### **利用可能なextras**
```toml
[tool.poetry.extras]
voice = ["speechrecognition", "pyaudio"]          # 音声認識
gui = ["PyQt5"]                                  # GUI機能
full = ["speechrecognition", "pyaudio", "PyQt5"]  # 全機能
```

### **選択的インストール**
```bash
# 音声認識のみ
poetry install --extras voice

# LLM機能も含める
poetry install --extras "voice llm"

# 全機能
poetry install --extras full

# 基本機能のみ（音声認識なし）
poetry install
```

## 🔧 トラブルシューティング

### **macOS PyAudioインストールエラー**
```bash
# Homebrewでportaudioをインストール
brew install portaudio

# 環境変数を設定してからインストール
export CPPFLAGS=-I/opt/homebrew/include
export LDFLAGS=-L/opt/homebrew/lib
poetry add pyaudio
```

### **M1/M2 Mac特有の問題**
```bash
# Apple Silicon用の設定
arch -arm64 brew install portaudio
poetry env use python3.11  # Python 3.11を使用
poetry install --extras voice
```

### **Linux（Ubuntu/Debian）の場合**
```bash
# システム依存関係
sudo apt-get install portaudio19-dev python3-pyaudio

# Poetry経由でインストール
poetry install --extras voice
```

## 🎯 動作確認

### **1. インストール確認**
```bash
poetry shell
python -c "
import speech_recognition as sr
import pyaudio
print('✅ 音声認識ライブラリ正常インストール')
print(f'SpeechRecognition: {sr.__version__}')
"
```

### **2. マイクテスト**
```bash
python -c "
import speech_recognition as sr
r = sr.Recognizer()
with sr.Microphone() as source:
    print('🎤 マイクテスト中...')
    r.adjust_for_ambient_noise(source)
    print('✅ マイク正常動作')
"
```

### **3. 音韻システム実行**
```bash
python -m app.examples.web_camera_oo
```

## 📊 依存関係の状況

### **必須依存関係**
```toml
python = "^3.10"
torchvision = "^0.16.2"
opencv-python = "^4.9.0.80"
streamdiffusion = "^0.1.1"
python-dotenv = "^1.0.0"
```

### **音声認識依存関係**
```toml
speechrecognition = "^3.10.0"  # Google音声認識API
pyaudio = "^0.2.11"            # マイク入力
```

### **オプション依存関係**
```toml
PyQt5 = "^5.15.0"             # GUI（将来の拡張用）
```

## 🎵 音韻システム使用例

### **実行後の動作**
1. **🎤 マイク起動**: 自動でマイク認識開始
2. **🗣️ 音声入力**: 「あさひ」と話す
3. **🎵 音韻抽出**: 音韻「あ」を抽出
4. **📝 プロンプト生成**: "morning sun, blue sky, autumn"
5. **🎨 画像生成**: StreamDiffusionで芸術作品生成

### **エラー時の動作**
```python
# ライブラリ未インストール時
❌ speech_recognitionライブラリがインストールされていません
✅ 音声認識なしで他機能は正常動作
```

## 🚀 完全セットアップコマンド

```bash
# 1. システム依存関係
brew install portaudio

# 2. Poetry環境セットアップ
poetry install --extras full

# 3. 環境変数設定（オプション）

# 4. 実行
poetry shell
python -m app.examples.web_camera_oo
```

## 📈 パフォーマンス最適化

### **高速化設定（自動適用）**
```python
# AppConfigで自動設定
optimize_for_speed = True     # 50%高速化
use_kohaku_model = True       # 芸術特化モデル
```

---

🎵 **Poetry環境で完璧な音韻トリガーシステムを楽しんでください！**