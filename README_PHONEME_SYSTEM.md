# 🎵 StreamDiffusion 音韻トリガーシステム

音韻をトリガーにしたリアルタイム画像生成システムです。日本語の最初の音韻を認識し、対応する英語プロンプトを自動生成してStreamDiffusionで画像を生成します。

## 🚀 機能概要

### 🎤 音韻トリガー機能
- **音声認識**: マイクから日本語音声を取得
- **音韻抽出**: 認識した日本語の最初の音韻を抽出
- **単語選択**: 音韻辞書から3つの英語表現をランダム選択
- **画像生成**: StreamDiffusionでリアルタイム画像生成

### 🎨 従来機能の強化
- **オブジェクト指向設計**: 拡張性・保守性の向上
- **モジュラー構造**: 各機能を独立したクラスで管理
- **エラーハンドリング**: 依存関係の欠如に対応
- **プロンプト履歴**: 生成したプロンプトの履歴管理

## 📋 必要な依存関係

### 必須
```bash
pip install opencv-python Pillow numpy torch diffusers streamdiffusion python-dotenv
```

### オプション（音声認識）
```bash
pip install SpeechRecognition pyaudio
```

## 🛠️ セットアップ

1. **環境変数設定**
```bash
```

2. **依存関係インストール**
```bash
pip install -r requirements.txt
```

## 🎯 使用方法

### オブジェクト指向版（推奨）
```bash
python -m app.examples.web_camera_oo
```

### 従来版
```bash
python -m app.examples.web_camera
```

## ⌨️ 操作方法

### 🎤 音声操作
- **話しかける**: マイクに向かって日本語で話すと自動で画像が変化
- **音韻の例**:
  - 「あさひ」→ 音韻「あ」→ morning sun, blue sky, autumn...
  - 「かぜ」→ 音韻「か」→ wind, river, radiance...
  - 「さくら」→ 音韻「さ」→ cherry blossom, fish, heron...

### ⌨️ キーボード操作
- `[r]` ランダムプロンプト生成
- `[i]` プロンプト入力モード
- `[s]` 現在の画像を保存
- `[p]` プロンプト履歴表示
- `[q]` 終了

### 🌐 リモート操作
```bash
# 別ターミナルから
python send_prompt.py
```

## 🏗️ アーキテクチャ

### クラス構造
```
WebCameraApp
├── PhonemeProcessor      # 音韻処理・辞書管理
├── VoiceRecognitionHandler  # 音声認識
├── PromptManager         # プロンプト管理・履歴
├── CameraController      # カメラ制御
├── NetworkServer         # TCP通信
├── UIController          # UI制御
└── StreamDiffusion       # 画像生成
```

### 音韻辞書システム
- **GOJUON_WORDS**: 50音別の単語・英語表現辞書
- **音韻抽出**: 日本語音声の最初の文字を音韻として使用
- **ランダム選択**: 各音韻から3つの英語表現をランダム選択
- **プロンプト合成**: 選択した表現を組み合わせて画像生成

## 🎵 音韻辞書例

```python
'あ': [('あさひ', 'morning sun'), ('あおぞら', 'blue sky'), ...],
'か': [('かぜ', 'wind'), ('かわ', 'river'), ...],
'さ': [('さくら', 'cherry blossom'), ('さかな', 'fish'), ...],
```

## 🔧 設定オプション

### AppConfig
```python
@dataclass
class AppConfig:
    sd_side_length: int = 512           # 画像サイズ
    tcp_port: int = 65432              # TCP通信ポート
    phoneme_max_prompts: int = 6       # 最大プロンプト数
    phoneme_update_interval: int = 30   # 更新間隔（秒）
    voice_recognition_enabled: bool = True  # 音声認識有効/無効
```

## 🐛 トラブルシューティング

### 音声認識が動作しない
```bash
# macOS
brew install portaudio
pip install pyaudio

# Ubuntu
sudo apt-get install portaudio19-dev
pip install pyaudio
```

### カメラが検出されない
- カメラのアクセス許可を確認
- 他のアプリケーションがカメラを使用していないか確認

### StreamDiffusion初期化エラー
- GPU/MPSデバイスの利用可能性を確認
- モデルファイルのダウンロード完了を確認

## 🎨 カスタマイズ

### 音韻辞書の拡張
`app/phoneme_dictionary.py`を編集して新しい単語・表現を追加

### プロンプト生成の調整
`PromptManager`クラスで生成ロジックをカスタマイズ

### UI・表示のカスタマイズ
`UIController`クラスで表示・操作をカスタマイズ

## 📈 今後の拡張予定

- [ ] 複数音韻の組み合わせ対応
- [ ] 音韻以外のトリガー（感情、テンポなど）
- [ ] WebUIでのリモート操作
- [ ] 音韻認識精度の向上
- [ ] プロンプト学習機能

---

🎵 **音韻が創り出すアートの世界をお楽しみください！**