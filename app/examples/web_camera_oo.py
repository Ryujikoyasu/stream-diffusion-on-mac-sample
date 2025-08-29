"""
StreamDiffusion Web Camera - Object-Oriented Refactored Version
音韻トリガーによるリアルタイム画像生成システム
"""

import time
import threading
import socket
import json
import random
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple
from queue import Queue
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

from ..stream_diffusion import StreamDiffusion
from ..phoneme_dictionary import GOJUON_WORDS

# 設定クラス
@dataclass
class AppConfig:
    """アプリケーション設定"""
    sd_side_length: int = 512
    output_image_file: str = "output.png"
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 65432
    gallery_dir: str = "gallery"
    max_history: int = 10
    voice_recognition_enabled: bool = True
    
    # 音韻システム設定
    phoneme_max_prompts: int = 6
    phoneme_update_interval: int = 12  # 12秒間隔
    max_selected_words: int = 3
    
    # StreamDiffusion最適化設定（ref版統合）
    optimize_for_speed: bool = True  # 音韻システムでは高速化を優先
    use_kohaku_model: bool = True    # 芸術特化モデルを使用
    local_cache_dir: str = "./models"

# 音韻プロセッサークラス
class PhonemeProcessor:
    """音韻処理と辞書管理"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.last_used_phoneme: Optional[str] = None
        
    def get_first_phoneme(self, text: str) -> str:
        """音声認識結果から最初の音韻を取得"""
        if not text:
            return ''
        return text[0]
    
    def get_random_words_from_phoneme(self, phoneme: str) -> List[str]:
        """音韻から単語をランダムに選択"""
        if phoneme not in GOJUON_WORDS:
            return []
            
        word_list = GOJUON_WORDS[phoneme]
        selected_count = min(self.config.max_selected_words, len(word_list))
        selected_pairs = random.sample(word_list, selected_count)
        return [pair[1] for pair in selected_pairs]  # 英語表現のリストを返す
    
    def process_voice_input(self, text: str) -> Tuple[str, List[str]]:
        """音声入力を処理して音韻と単語を返す"""
        if not text:
            return "", []
            
        # テキストから最初の2文字分の音韻を取得
        first_two_phonemes = [char for char in text[:2] if char in GOJUON_WORDS]
        if not first_two_phonemes:
            return "", []
        
        # 音韻の選択（前回と同じ場合は2文字目を使用）
        selected_phoneme = first_two_phonemes[0]
        if (self.last_used_phoneme and 
            selected_phoneme == self.last_used_phoneme and 
            len(first_two_phonemes) > 1):
            selected_phoneme = first_two_phonemes[1]
        
        self.last_used_phoneme = selected_phoneme
        english_words = self.get_random_words_from_phoneme(selected_phoneme)
        
        return selected_phoneme, english_words

# 音声認識ハンドラー
class VoiceRecognitionHandler:
    """音声認識処理"""
    
    def __init__(self, config: AppConfig, phoneme_processor: PhonemeProcessor):
        self.config = config
        self.phoneme_processor = phoneme_processor
        self.recognition_active = config.voice_recognition_enabled
        self.running = False
        self.voice_callback: Optional[Callable] = None
        
        if self.recognition_active:
            self._initialize_voice_recognition()
    
    def _initialize_voice_recognition(self):
        """音声認識の初期化"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = 4000
            print("🎤 音声認識の初期化が完了しました")
        except ImportError:
            print("❌ speech_recognitionライブラリがインストールされていません")
            self.recognition_active = False
        except Exception as e:
            print(f"❌ 音声認識の初期化に失敗: {e}")
            self.recognition_active = False
    
    def set_voice_callback(self, callback: Callable[[str, List[str]], None]):
        """音声認識結果のコールバック設定"""
        self.voice_callback = callback
    
    def start_listening(self):
        """音声認識開始"""
        if not self.recognition_active:
            return
            
        self.running = True
        self.voice_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.voice_thread.start()
        print("🎤 音声認識を開始しました")
    
    def stop_listening(self):
        """音声認識停止"""
        self.running = False
    
    def _listen_loop(self):
        """音声認識ループ"""
        import speech_recognition as sr
        
        while self.running:
            try:
                with sr.Microphone() as source:
                    if not hasattr(self, 'first_adjustment'):
                        print("🔧 マイクのノイズ調整中...")
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                        self.first_adjustment = True
                    
                    print("🎤 音声認識待機中...")
                    audio = self.recognizer.listen(source, phrase_time_limit=3)
                    text = self.recognizer.recognize_google(audio, language='ja-JP')
                    print(f"🗣️ 認識された音声: {text}")
                    
                    # 音韻処理
                    phoneme, words = self.phoneme_processor.process_voice_input(text)
                    if words and self.voice_callback:
                        self.voice_callback(phoneme, words)
                        
            except sr.UnknownValueError:
                pass  # 認識できない場合は無視
            except sr.RequestError as e:
                print(f"❌ 音声認識サービスエラー: {e}")
            except Exception as e:
                print(f"❌ 音声認識エラー: {e}")
                time.sleep(1)

# プロンプト管理クラス
class PromptManager:
    """プロンプトの生成・管理・履歴"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.current_prompt: str = ""
        self.prompt_history: List[str] = []
        self.active_prompts: List[str] = []
        self.last_update_time = time.time()
        
        # LLM API設定
        load_dotenv()
        self.llm_available = False
        
        # 初期プロンプト設定（StreamDiffusionのデフォルトに合わせる）
        self.default_prompt = "moon"
        self.base_negative_prompt = ("human, person, people, face, portrait, "
                                   "low quality, bad quality, blurry, low resolution")
    
    def _check_llm_availability(self) -> bool:
        """LLM APIの利用可能性をチェック"""
        return False
    
    def generate_random_prompt(self) -> str:
        """ランダムプロンプト生成"""
        themes = [
            "moonlit landscape", "impressionist garden", "watercolor nature",
            "serene mountain view", "peaceful lakeside", "autumn forest",
            "gentle sunrise", "quiet village scene", "classical still life",
            "pastoral countryside", "misty morning", "soft floral arrangement"
        ]
        
        theme = random.choice(themes)
        
        # LLMが利用できない場合は基本的なプロンプトを返す
        if not self.llm_available:
            return f"{theme}, impressionist painting style, soft brushstrokes, gentle lighting"
        
        return f"{theme}, impressionist painting style, soft brushstrokes, gentle lighting, serene atmosphere"
    
    def enhance_prompt(self, user_prompt: str) -> str:
        """プロンプト強化"""
        if not user_prompt.strip():
            return self.generate_random_prompt()
        
        # Simple enhancement without LLM
        enhanced = f"{user_prompt}, impressionist painting style, soft colors, gentle lighting, serene atmosphere"
        return enhanced
    
    def add_phoneme_prompts(self, phoneme: str, words: List[str]):
        """音韻ベースのプロンプトを追加"""
        if not words:
            return
        
        # 重複除去と上限管理
        new_words = [w for w in words if w not in self.active_prompts]
        self.active_prompts.extend(new_words)
        
        if len(self.active_prompts) > self.config.phoneme_max_prompts:
            excess = len(self.active_prompts) - self.config.phoneme_max_prompts
            self.active_prompts = self.active_prompts[excess:]
        
        # プロンプト生成
        combined_prompt = f"{self.default_prompt}, {', '.join(self.active_prompts)}"
        self.update_current_prompt(combined_prompt)
        
        print(f"🎵 音韻「{phoneme}」→ {', '.join(words)}")
        print(f"📝 現在のプロンプト: {combined_prompt}")
    
    def update_current_prompt(self, prompt: str):
        """現在のプロンプトを更新"""
        self.current_prompt = prompt
        self.add_to_history(prompt)
        self.last_update_time = time.time()
        # プロンプト切り替えをターミナルに出力
        print(f"🎨 プロンプト更新: {prompt}")
    
    def add_to_history(self, prompt: str):
        """履歴に追加"""
        if len(self.prompt_history) >= self.config.max_history:
            self.prompt_history.pop(0)
        self.prompt_history.append(prompt)
    
    def can_update_prompt(self) -> bool:
        """プロンプト更新可能かチェック"""
        return (time.time() - self.last_update_time) >= self.config.phoneme_update_interval

# カメラコントローラークラス  
class CameraController:
    """カメラ制御"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_id: Optional[int] = None
    
    def find_available_cameras(self) -> List[int]:
        """利用可能なカメラを検出"""
        available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def select_camera(self) -> Optional[int]:
        """カメラ選択"""
        available_cameras = self.find_available_cameras()
        
        if not available_cameras:
            print("❌ 利用可能なカメラが見つかりませんでした")
            return None
        
        print(f"🎥 利用可能なカメラ: {available_cameras}")
        
        if len(available_cameras) == 1:
            camera_id = available_cameras[0]
            print(f"📹 カメラ {camera_id} を使用します")
            return camera_id
        
        while True:
            try:
                choice = input(f"使用するカメラを選択してください {available_cameras}: ")
                camera_id = int(choice)
                if camera_id in available_cameras:
                    return camera_id
                print(f"❌ 無効な選択です。{available_cameras} から選択してください")
            except ValueError:
                print("❌ 数字を入力してください")
    
    def initialize_camera(self) -> bool:
        """カメラ初期化"""
        self.camera_id = self.select_camera()
        if self.camera_id is None:
            return False
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"❌ カメラ {self.camera_id} を開けませんでした")
            return False
        
        return True
    
    def read_frame(self) -> Optional[np.ndarray]:
        """フレーム読み取り"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        """カメラリソース解放"""
        if self.cap:
            self.cap.release()

# ネットワークサーバークラス
class NetworkServer:
    """TCP通信サーバー"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.prompt_callback: Optional[Callable] = None
        
    def set_prompt_callback(self, callback: Callable[[str], None]):
        """プロンプト受信コールバック設定"""
        self.prompt_callback = callback
    
    def start_server(self):
        """サーバー開始"""
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        print(f"🌐 TCPサーバーを開始: {self.config.tcp_host}:{self.config.tcp_port}")
    
    def stop_server(self):
        """サーバー停止"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
    
    def _server_loop(self):
        """サーバーループ"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.config.tcp_host, self.config.tcp_port))
                s.listen()
                self.server_socket = s
                
                while self.running:
                    try:
                        conn, addr = s.accept()
                        self._handle_client(conn, addr)
                    except Exception as e:
                        if self.running:
                            print(f"❌ サーバーエラー: {e}")
                            
        except Exception as e:
            print(f"❌ サーバー初期化エラー: {e}")
    
    def _handle_client(self, conn: socket.socket, addr):
        """クライアント処理"""
        print(f"🔗 クライアント接続: {addr}")
        try:
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    
                    prompt = data.decode('utf-8').strip()
                    print(f"📨 受信プロンプト: {prompt}")
                    
                    if self.prompt_callback:
                        self.prompt_callback(prompt)
                    
                    conn.sendall(f"✅ プロンプト受信: {prompt}".encode('utf-8'))
        except Exception as e:
            print(f"❌ クライアント処理エラー: {e}")

# ユーザーインターフェースコントローラー
class UIController:
    """ユーザーインターフェース制御"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.last_output_image: Optional[Image.Image] = None
        
        # 画像履歴管理（軽量化のため最大3枚まで）
        self.image_history: List[Image.Image] = []
        self.history_size = 3
        
        # ウィンドウ管理
        self.window_name = "StreamDiffusion 音韻トリガー"
        self.window_initialized = False
        
    def show_instructions(self):
        """操作説明表示"""
        print("\n" + "="*45)
        print("🎨 StreamDiffusion 音韻トリガーシステム")
        print("="*45)
        print("🎤 マイクに向かって日本語で話してください")
        print("🎵 最初の音韻に対応した画像が生成されます")
        print(f"🌐 TCP接続: {self.config.tcp_host}:{self.config.tcp_port}")
        print("\n⌨️  キーボード操作:")
        print("[r] ランダムプロンプト生成")
        print("[i] プロンプト入力モード") 
        print("[s] 現在の画像を保存")
        print("[p] プロンプト履歴表示")
        print("[c] プロンプトリセット（moonのみ）")
        print("[q] 終了")
        print("="*45 + "\n")
    
    def display_frame(self, output_image: Image.Image):
        """フレーム表示（履歴管理付き・全画面表示・中央寄せ）"""
        # 履歴に追加
        self._add_to_history(output_image)
        
        # 透明度で重ね合わせた画像を作成
        blended_image = self._create_blended_image()
        
        # 中央寄せ用の画像を作成（16:9比率）
        centered_image = self._create_centered_display(blended_image)
        
        self.last_output_image = blended_image
        output_np = cv2.cvtColor(np.array(centered_image), cv2.COLOR_RGB2BGR)
        
        # 初回のみウィンドウを作成・設定
        if not self.window_initialized:
            height, width = output_np.shape[:2]
            print(f"🔍 デバッグ - 画像サイズ: {width}x{height}")
            
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # 画像サイズに強制リサイズしてから全画面化
            cv2.resizeWindow(self.window_name, width, height)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            self.window_initialized = True
        
        # 画像のみ更新
        cv2.imshow(self.window_name, output_np)
    
    def _add_to_history(self, image: Image.Image):
        """画像を履歴に追加"""
        # 新しい画像を先頭に追加
        self.image_history.insert(0, image.copy())
        
        # 履歴サイズを制限
        if len(self.image_history) > self.history_size:
            self.image_history = self.image_history[:self.history_size]
    
    def _create_blended_image(self) -> Image.Image:
        """透明度を使って画像を重ね合わせ"""
        if not self.image_history:
            # 黒い画像を返す
            return Image.new('RGB', (512, 512), color=(0, 0, 0))
        
        # 透明度設定
        alphas = [0.6, 0.2, 0.1]  # t, t-1, t-2の透明度
        
        # ベース画像（最新）
        result = self.image_history[0].copy().convert('RGBA')
        
        # 古い画像を重ね合わせ
        for i in range(1, min(len(self.image_history), len(alphas))):
            alpha = alphas[i]
            old_image = self.image_history[i].convert('RGBA')
            
            # アルファブレンディング
            overlay = Image.new('RGBA', result.size, (0, 0, 0, 0))
            overlay.paste(old_image, (0, 0))
            
            # 透明度を適用
            overlay.putalpha(int(255 * alpha))
            
            # 重ね合わせ
            result = Image.alpha_composite(result, overlay)
        
        # RGBに変換して返す
        return result.convert('RGB')
    
    def _create_centered_display(self, image: Image.Image) -> Image.Image:
        """正方形画像を画面中央に配置（上下左右黒色）"""
        # M4 MacBook Pro解像度に固定
        display_width = 2560
        display_height = 1664
        
        # 正方形画像のサイズ（画面の短辺に合わせる）
        square_size = min(display_width, display_height)  # 1664x1664
        
        # 黒い背景を作成
        display_image = Image.new('RGB', (display_width, display_height), color=(0, 0, 0))
        
        # 正方形画像をリサイズ
        resized_image = image.resize((square_size, square_size), Image.LANCZOS)
        
        # 中央に配置（上下左右完全中央）
        x_offset = (display_width - square_size) // 2  # 水平中央
        y_offset = (display_height - square_size) // 2  # 垂直中央
        
        display_image.paste(resized_image, (x_offset, y_offset))
        
        return display_image
    
    def save_to_gallery(self, image: Image.Image, prompt: str) -> str:
        """ギャラリーに保存"""
        os.makedirs(self.config.gallery_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.gallery_dir}/img_{timestamp}.png"
        
        image.save(filename)
        with open(f"{self.config.gallery_dir}/img_{timestamp}.json", "w") as f:
            json.dump({"prompt": prompt, "timestamp": timestamp}, f)
        
        return filename

# メインアプリケーションクラス
class WebCameraApp:
    """メインアプリケーション"""
    
    def __init__(self):
        self.config = AppConfig()
        
        # コンポーネント初期化
        self.phoneme_processor = PhonemeProcessor(self.config)
        self.prompt_manager = PromptManager(self.config)
        self.camera_controller = CameraController(self.config)
        self.voice_handler = VoiceRecognitionHandler(self.config, self.phoneme_processor)
        self.network_server = NetworkServer(self.config)
        self.ui_controller = UIController(self.config)
        
        # StreamDiffusion
        self.stream_diffusion: Optional[StreamDiffusion] = None
        
        # コールバック設定
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """コールバック設定"""
        self.voice_handler.set_voice_callback(self._on_voice_input)
        self.network_server.set_prompt_callback(self._on_network_prompt)
    
    def _on_voice_input(self, phoneme: str, words: List[str]):
        """音声入力コールバック"""
        if self.prompt_manager.can_update_prompt():
            self.prompt_manager.add_phoneme_prompts(phoneme, words)
            self._update_stream_diffusion()  # StreamDiffusion更新を追加
    
    def _on_network_prompt(self, prompt: str):
        """ネットワークプロンプトコールバック"""
        enhanced_prompt = self.prompt_manager.enhance_prompt(prompt)
        self.prompt_manager.update_current_prompt(enhanced_prompt)
        self._update_stream_diffusion()  # StreamDiffusion更新を追加
    
    def initialize(self) -> bool:
        """アプリケーション初期化"""
        print("🚀 アプリケーション初期化中...")
        
        # カメラ初期化
        if not self.camera_controller.initialize_camera():
            print("❌ カメラ初期化に失敗しました")
            return False
        
        # StreamDiffusion初期化
        initial_prompt = self.prompt_manager.default_prompt
        # 音韻プロンプトリストをクリア
        self.prompt_manager.active_prompts = []
        self.prompt_manager.update_current_prompt(initial_prompt)
        print(f"🎨 初期プロンプト設定: {initial_prompt}")
        
        try:
            self.stream_diffusion = StreamDiffusion(
                prompt=initial_prompt,
                local_cache_dir=self.config.local_cache_dir,
                optimize_for_speed=self.config.optimize_for_speed,
                use_kohaku_model=self.config.use_kohaku_model
            )
            print("✅ StreamDiffusion初期化完了（ref版最適化適用）")
            
            # 初期化後に明示的にプロンプトを設定
            self._update_stream_diffusion()
        except Exception as e:
            print(f"❌ StreamDiffusion初期化エラー: {e}")
            return False
        
        # 音声認識開始
        self.voice_handler.start_listening()
        
        # ネットワークサーバー開始
        self.network_server.start_server()
        
        return True
    
    def run(self):
        """メインループ実行"""
        if not self.initialize():
            return
        
        self.ui_controller.show_instructions()
        
        try:
            while True:
                # フレーム取得
                frame = self.camera_controller.read_frame()
                if frame is None:
                    print("❌ カメラからフレームを取得できませんでした")
                    break
                
                # 画像処理
                init_img = self._preprocess_frame(frame)
                image_tensor = self.stream_diffusion.stream.preprocess_image(init_img)
                
                try:
                    output_image = self.stream_diffusion.stream(image_tensor)
                    if isinstance(output_image, Image.Image):
                        self.ui_controller.display_frame(output_image)
                except Exception as e:
                    print(f"❌ 画像生成エラー: {e}")
                    if self.ui_controller.last_output_image:
                        self.ui_controller.display_frame(self.ui_controller.last_output_image)
                
                # キー入力処理
                if not self._handle_keyboard_input():
                    break
                    
        except KeyboardInterrupt:
            print("👋 キーボード割り込みで終了")
        except Exception as e:
            print(f"❌ 実行エラー: {e}")
        finally:
            self._cleanup()
    
    def _preprocess_frame(self, frame: np.ndarray) -> Image.Image:
        """フレーム前処理"""
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self._crop_center(
            pil_frame, 
            self.config.sd_side_length * 2, 
            self.config.sd_side_length * 2
        ).resize((self.config.sd_side_length, self.config.sd_side_length), Image.NEAREST)
    
    def _crop_center(self, pil_img: Image.Image, crop_width: int, crop_height: int) -> Image.Image:
        """中央クロップ"""
        img_width, img_height = pil_img.size
        return pil_img.crop((
            (img_width - crop_width) // 2,
            (img_height - crop_height) // 2,
            (img_width + crop_width) // 2,
            (img_height + crop_height) // 2,
        ))
    
    def _handle_keyboard_input(self) -> bool:
        """キーボード入力処理"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("👋 終了します")
            return False
        elif key == ord('r'):
            new_prompt = self.prompt_manager.generate_random_prompt()
            self.prompt_manager.update_current_prompt(new_prompt)
            self._update_stream_diffusion()
            print(f"🎲 ランダムプロンプト: {new_prompt}")
        elif key == ord('i'):
            self._interactive_prompt_input()
        elif key == ord('s') and self.ui_controller.last_output_image:
            filename = self.ui_controller.save_to_gallery(
                self.ui_controller.last_output_image, 
                self.prompt_manager.current_prompt
            )
            print(f"💾 画像保存: {filename}")
        elif key == ord('p'):
            self._show_prompt_history()
        elif key == ord('c'):
            self._reset_to_default_prompt()
        
        return True
    
    def _interactive_prompt_input(self):
        """インタラクティブプロンプト入力"""
        print("\n💡 プロンプト入力モード")
        try:
            user_input = input("新しいプロンプト (空白でランダム): ").strip()
            
            if user_input == "":
                new_prompt = self.prompt_manager.generate_random_prompt()
                print(f"🎲 ランダムプロンプト: {new_prompt}")
            else:
                new_prompt = self.prompt_manager.enhance_prompt(user_input)
                print(f"✨ 強化プロンプト: {new_prompt}")
            
            self.prompt_manager.update_current_prompt(new_prompt)
            self._update_stream_diffusion()
            
        except Exception as e:
            print(f"❌ プロンプト入力エラー: {e}")
    
    def _show_prompt_history(self):
        """プロンプト履歴表示"""
        print("\n📚 プロンプト履歴:")
        for i, prompt in enumerate(self.prompt_manager.prompt_history, 1):
            print(f"{i}. {prompt}")
        print()
    
    def _reset_to_default_prompt(self):
        """プロンプトをデフォルト（moon）にリセット"""
        self.prompt_manager.active_prompts = []
        self.prompt_manager.update_current_prompt(self.prompt_manager.default_prompt)
        self._update_stream_diffusion()
        print(f"🔄 プロンプトリセット: {self.prompt_manager.default_prompt}")
    
    def _update_stream_diffusion(self):
        """StreamDiffusion更新"""
        try:
            self.stream_diffusion.stream.prepare(
                prompt=self.prompt_manager.current_prompt,
                negative_prompt=self.prompt_manager.base_negative_prompt,
                num_inference_steps=30,  # 50→30に減らして高速化
                guidance_scale=2.5,     # 1.2→2.5に上げてクリエイティブに
                delta=0.3,              # 0.2→0.3に上げてクリエイティブさと原型保持のバランス
            )
        except Exception as e:
            print(f"❌ StreamDiffusion更新エラー: {e}")
    
    def _cleanup(self):
        """リソース解放"""
        print("🧹 リソース解放中...")
        self.voice_handler.stop_listening()
        self.network_server.stop_server()
        self.camera_controller.release()
        cv2.destroyAllWindows()
        print("✅ 終了完了")

def main():
    """メイン関数"""
    app = WebCameraApp()
    app.run()

if __name__ == "__main__":
    main()