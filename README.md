# stream-diffusion-on-mac-sample

StreamDiffusion を macOS で動かすサンプル

↓実行結果 (3 fps くらい)

![output](https://github.com/kawamou/stream-diffusion-on-mac-sample/assets/18514782/5cf4ebab-f1a9-4a10-99c3-810db1df5198)


## 実行方法

```sh
poetry shell
poetry install
```

`.venv/lib/python3.XX/site-packages/streamdiffusion`以下の`pipeline.py`の`cuda`依存部分を修正

<img width="1075" alt="image" src="https://github.com/kawamou/stream-diffusion-on-mac-sample/assets/18514782/c60c7252-0076-4a49-bcff-a932f2e04bdd">

修正後実行

```sh
python -m app.examples.web-camera
```

## 設定ファイル

プロジェクトは `config.json` で設定を管理しています。主な設定項目：

### ディスプレイ設定
- `width`, `height`: 画面サイズ (デフォルト: 800x800)
- `fps`: フレームレート (デフォルト: 60)
- `display_width`, `display_height`: 表示ウィンドウサイズ (デフォルト: 2048x2048)

### StreamDiffusion設定  
- `guidance_scale`: クリエイティビティ制御 (低いほど自由, デフォルト: 0.6)
- `delta`: 変化の大きさ (高いほど大胆, デフォルト: 1.5)
- `use_random_seed`: ランダムシード使用 (デフォルト: true)

### クリエイティビティ設定
- `creativity_update_interval`: クリエイティブ要素更新間隔 (デフォルト: 30フレーム)
- `max_frame_history`: フレーム履歴数 (デフォルト: 3)
- `themes`: 初期テーマリスト
- `creative_modifiers`: ランダム修飾詞リスト

設定変更は `config.json` を編集して適用できます。
