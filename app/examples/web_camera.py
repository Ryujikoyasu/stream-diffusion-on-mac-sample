import time
from typing import Any, Callable
import socket
import threading
import random
import os
import json
from datetime import datetime
import pickle

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

from ..stream_diffusion import StreamDiffusion

# Load environment variables
load_dotenv()

SD_SIDE_LENGTH = 512
OUTPUT_IMAGE_FILE = "output.png"
HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

# フレーム受信設定
FRAME_HOST = "127.0.0.1"
FRAME_PORT = 65433  # main_moon.pyからのフレーム受信用

# Themes for initial prompts
THEMES = [
    "nature and wildlife",
    "cyberpunk city",
    "fantasy creatures",
    "space exploration",
    "underwater world",
    "steampunk invention",
    "magical forest",
    "futuristic technology"
]

# Random creative modifiers for variety
CREATIVE_MODIFIERS = [
    "vibrant colors", "soft pastels", "neon glow", "watercolor style", "oil painting",
    "abstract expressionism", "surreal", "dreamy atmosphere", "crystalline structures",
    "flowing liquid", "particle effects", "light rays", "ethereal mist", "iridescent",
    "holographic", "prismatic", "luminous", "cosmic energy", "electric blue",
    "golden hour", "moonlit", "aurora colors", "fractal patterns", "organic shapes"
]

# Image history for smooth transitions (limited to 3 frames)
FRAME_HISTORY = []
MAX_FRAME_HISTORY = 3

# History of successful prompts
PROMPT_HISTORY = []
MAX_HISTORY = 10

# フレームブレンド設定
FRAME_BLEND_ALPHA = 0.3  # 新フレームの影響率

# Path for saving gallery images
GALLERY_DIR = "gallery"
os.makedirs(GALLERY_DIR, exist_ok=True)

# ウィンドウ表示設定
WINDOW_NAME = "StreamDiffusion"
DISPLAY_WIDTH = 2048  # 表示用ウィンドウサイズを2倍に
DISPLAY_HEIGHT = 2048
window_initialized = False


def generate_random_prompt():
    """Generate a random initial prompt based on a theme"""
    theme = random.choice(THEMES)
    base_modifiers = ["colorful", "detailed", "artistic"]
    extra_modifier = random.choice(CREATIVE_MODIFIERS)
    return f"{theme}, {', '.join(base_modifiers)}, {extra_modifier}"

def add_creative_randomness(prompt):
    """Add random creative modifiers for variety"""
    modifiers = random.sample(CREATIVE_MODIFIERS, min(3, len(CREATIVE_MODIFIERS)))
    return f"{prompt}, {', '.join(modifiers)}"

def enhance_prompt(user_prompt, current_prompt=None, append_mode=False):
    """Enhance user prompt with basic keywords and random creativity"""
    if not user_prompt.strip():
        base_prompt = generate_random_prompt()
        return add_creative_randomness(base_prompt)
    
    if append_mode and current_prompt:
        # 追加モード：既存プロンプトに新しい要素を追加
        enhanced = f"{current_prompt}, {user_prompt}"
    else:
        # 新規モード：完全に新しいプロンプト
        enhanced = f"{user_prompt}, detailed, high quality, artistic"
        enhanced = add_creative_randomness(enhanced)
    
    return enhanced

def add_text_to_image(image, text, position="bottom"):
    """Add text overlay to image"""
    draw = ImageDraw.Draw(image)
    
    # Try to use a nicer font if available, otherwise use default
    try:
        font = ImageFont.truetype("Arial", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Wrap text to fit image width
    text_width = SD_SIDE_LENGTH - 20
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        if draw.textlength(test_line, font=font) <= text_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    
    text_height = len(lines) * 25
    
    # Position text based on parameter
    if position == "bottom":
        y_position = image.height - text_height - 10
    elif position == "top":
        y_position = 10
    else:  # center
        y_position = (image.height - text_height) // 2
    
    # Add semi-transparent background for text
    for i, line in enumerate(lines):
        line_y = y_position + i * 25
        text_width = draw.textlength(line, font=font)
        draw.rectangle([(10, line_y - 5), (10 + text_width, line_y + 20)], fill=(0, 0, 0, 128))
        draw.text((10, line_y), line, fill=(255, 255, 255), font=font)
    
    return image

def save_to_gallery(image, prompt):
    """Save the generated image to gallery with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{GALLERY_DIR}/img_{timestamp}.png"
    image.save(filename)
    
    # Save prompt metadata
    with open(f"{GALLERY_DIR}/img_{timestamp}.json", "w") as f:
        json.dump({"prompt": prompt, "timestamp": timestamp}, f)
    
    return filename

def run_server(stream, enhance_fn):
    """Run TCP server to receive prompts and process them with LLM"""
    global PROMPT_HISTORY, current_prompt, waiting_for_new_image
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Listening for prompts on {HOST}:{PORT}")
        
        while True:
            conn, addr = s.accept()  # This blocks until a connection is made
            with conn:
                print(f"Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                        
                    # Get raw user prompt
                    user_prompt = data.decode("utf-8").strip()
                    print(f"Received user prompt: {user_prompt}")
                    
                    if user_prompt:
                        # Enhance prompt with LLM
                        enhanced_prompt = enhance_fn(user_prompt)
                        print(f"Enhanced prompt: {enhanced_prompt}")
                        
                        # Update current prompt and history
                        current_prompt = enhanced_prompt
                        if len(PROMPT_HISTORY) >= MAX_HISTORY:
                            PROMPT_HISTORY.pop(0)  # Remove oldest prompt
                        PROMPT_HISTORY.append(enhanced_prompt)
                        
                        # Signal that we're waiting for a new image
                        waiting_for_new_image = True
                        
                        # Update stream with the enhanced prompt
                        stream.prepare(
                            prompt=enhanced_prompt,
                            negative_prompt="low quality, bad quality, blurry, low resolution",
                            num_inference_steps=25,  # リアルタイム性向上
                            guidance_scale=0.6,      # クリエイティビティ向上
                            delta=1.5,               # 変化を大きく
                        )
                        
                        # Send acknowledgment back to client
                        conn.sendall(f"Processing prompt: {enhanced_prompt}".encode("utf-8"))

def run_controls(stream, enhance_fn):
    """Thread to handle keyboard controls"""
    global PROMPT_HISTORY, current_prompt, waiting_for_new_image, LAST_OUTPUT_IMAGE
    
    import keyboard
    
    # Try to import keyboard, but handle case where it's not available
    try:
        import keyboard
        keyboard_available = True
    except ImportError:
        print("Keyboard module not available. Install with: pip install keyboard")
        keyboard_available = False
        return
    
    if not keyboard_available:
        return
        
    def on_key_r():
        """Generate new random prompt"""
        global current_prompt, waiting_for_new_image
        print("Generating new random prompt...")
        new_prompt = generate_random_prompt()
        print(f"New random prompt: {new_prompt}")
        current_prompt = new_prompt
        waiting_for_new_image = True
        
        # Update prompt history
        if len(PROMPT_HISTORY) >= MAX_HISTORY:
            PROMPT_HISTORY.pop(0)  # Remove oldest prompt
        PROMPT_HISTORY.append(new_prompt)
        
        # Update stream with new prompt
        stream.prepare(
            prompt=new_prompt,
            negative_prompt="low quality, bad quality, blurry, low resolution",
            num_inference_steps=25,
            guidance_scale=0.6,
            delta=1.5,
        )
    
    def on_key_s():
        """Save current image to gallery"""
        if FRAME_HISTORY:
            filename = save_to_gallery(FRAME_HISTORY[-1], current_prompt)
            print(f"Saved current image to {filename}")
    
    def on_key_p():
        """Print prompt history"""
        print("\n=== Prompt History ===")
        for i, prompt in enumerate(PROMPT_HISTORY):
            print(f"{i+1}. {prompt}")
        print("====================\n")
        
    # Register keyboard hooks
    keyboard.add_hotkey('r', on_key_r)
    keyboard.add_hotkey('s', on_key_s)
    keyboard.add_hotkey('p', on_key_p)
    
    # Keep thread alive
    keyboard.wait("q")

def timeit(func: Callable[..., Any]):
    def wrapper(*args, **kwargs) -> tuple[Image.Image, str]:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start
        return result, f"{1 / elapsed_time}"

    return wrapper


def crop_center(pil_img: Image.Image, crop_width: int, crop_height: int):
    img_width, img_height = pil_img.size
    return pil_img.crop(
        (
            (img_width - crop_width) // 2,
            (img_height - crop_height) // 2,
            (img_width + crop_width) // 2,
            (img_height + crop_height) // 2,
        )
    )


# フレーム受信クラス（再接続対応）
class FrameReceiver:
    def __init__(self, host=FRAME_HOST, port=FRAME_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self.latest_frame = None
        self.running = False
        self.connected = False
        
    def connect_to_sender(self):
        """フレーム送信元に接続（初回のみ）"""
        return self._try_connect()
    
    def _try_connect(self):
        """接続を試行"""
        try:
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"✨ main_moon.pyに接続成功: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ main_moon.pyへの接続失敗: {e}")
            self.connected = False
            return False
    
    def start_receiving(self):
        """フレーム受信開始"""
        self.running = True
        self.receive_thread = threading.Thread(target=self._receive_loop_with_reconnect, daemon=True)
        self.receive_thread.start()
        return True
    
    def _receive_loop_with_reconnect(self):
        """フレーム受信ループ（再接続対応）"""
        while self.running:
            if not self.connected:
                print("🔄 main_moon.pyへの再接続を試行中...")
                if self._try_connect():
                    print("✨ 再接続成功")
                else:
                    time.sleep(3)  # 3秒待ってから再試行
                    continue
            
            try:
                self._receive_frames()
            except Exception as e:
                print(f"❌ フレーム受信エラー: {e}")
                print("🔌 接続が切断されました。再接続を試行します...")
                self.connected = False
                time.sleep(2)
    
    def _receive_frames(self):
        """フレーム受信メインループ"""
        while self.running and self.connected:
            try:
                # サイズを受信（4バイト）
                size_data = self._recv_all(4)
                if not size_data:
                    raise ConnectionError("サイズデータの受信に失敗")
                
                # キープアライブチェック
                if size_data == b'\x00\x00\x00\x00':
                    continue  # キープアライブなのでスキップ
                    
                frame_size = int.from_bytes(size_data, byteorder='big')
                
                # フレームデータを受信
                frame_data = self._recv_all(frame_size)
                if not frame_data:
                    raise ConnectionError("フレームデータの受信に失敗")
                
                # pickleでデシリアライズ
                frame_array = pickle.loads(frame_data)
                self.latest_frame = frame_array
                
            except Exception as e:
                raise e
    
    def _recv_all(self, size):
        """指定サイズのデータを全て受信"""
        data = b''
        while len(data) < size:
            chunk = self.socket.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def get_latest_frame(self):
        """最新フレームを取得"""
        return self.latest_frame
    
    def stop_receiving(self):
        """受信停止"""
        self.running = False
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass

def find_available_cameras():
    """利用可能なカメラデバイスを検出する"""
    available_cameras = []
    for i in range(10):  # 0-9まで検査
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras


def select_input_source():
    """入力ソースを選択（カメラ or main_moon.py）"""
    print("🎥 入力ソースを選択してください:")
    print("1. カメラ")
    print("2. main_moon.pyのフレーム")
    
    while True:
        try:
            choice = input("選択 (1 or 2): ").strip()
            if choice == "1":
                return "camera"
            elif choice == "2":
                return "moon_frames"
            else:
                print("❌ 1 または 2 を入力してください")
        except ValueError:
            print("❌ 1 または 2 を入力してください")

def select_camera():
    """ユーザーにカメラを選択させる"""
    available_cameras = find_available_cameras()
    
    if not available_cameras:
        print("❌ 利用可能なカメラが見つかりませんでした")
        return None
    
    print(f"🎥 利用可能なカメラ: {available_cameras}")
    
    if len(available_cameras) == 1:
        print(f"📹 カメラ {available_cameras[0]} を使用します")
        return available_cameras[0]
    
    while True:
        try:
            choice = input(f"使用するカメラを選択してください {available_cameras}: ")
            camera_id = int(choice)
            if camera_id in available_cameras:
                return camera_id
            else:
                print(f"❌ 無効な選択です。{available_cameras} から選択してください")
        except ValueError:
            print("❌ 数字を入力してください")


def prompt_input_thread(stream, enhance_fn):
    """別スレッドでプロンプト入力を受け付ける"""
    global current_prompt, PROMPT_HISTORY
    
    print("💡 プロンプト入力スレッドを開始しました")
    print("プロンプト入力方法:")
    print("  + [テキスト] : 現在のプロンプトに追加")
    print("  [テキスト]   : 新しいプロンプトに置換")
    print("  (空白)       : ランダムプロンプト")
    
    while True:
        try:
            user_input = input("プロンプト入力: ").strip()
            
            if user_input.lower() == 'quit' or user_input.lower() == 'exit':
                print("👋 プロンプト入力を終了します")
                break
                
            if user_input == "":
                # 空白の場合はランダムプロンプト生成
                new_prompt = generate_random_prompt()
                print(f"🎲 ランダムプロンプト: {new_prompt}")
            elif user_input.startswith("+"):
                # 追加モード
                addition = user_input[1:].strip()
                if addition:
                    new_prompt = enhance_fn(addition, current_prompt, append_mode=True)
                    print(f"➕ 追加されたプロンプト: {new_prompt}")
                else:
                    print("❌ 追加する内容が空です")
                    continue
            else:
                # 通常入力は追加モード（累積）
                new_prompt = enhance_fn(user_input, current_prompt, append_mode=True)
                print(f"➕ 追加されたプロンプト: {new_prompt}")
            
            # グローバル変数更新
            current_prompt = new_prompt
            
            # プロンプト履歴に追加
            if len(PROMPT_HISTORY) >= MAX_HISTORY:
                PROMPT_HISTORY.pop(0)
            PROMPT_HISTORY.append(new_prompt)
            
            # ストリームを更新
            stream.prepare(
                prompt=new_prompt,
                negative_prompt="low quality, bad quality, blurry, low resolution",
                num_inference_steps=20,
                guidance_scale=0.5,
                delta=2.0,
            )
            
            print(f"🔄 プロンプトを更新しました: {new_prompt}")
            
        except KeyboardInterrupt:
            print("👋 プロンプト入力を終了します")
            break
        except Exception as e:
            print(f"❌ プロンプト処理エラー: {e}")
            continue



def main():
    global FRAME_HISTORY, PROMPT_HISTORY, current_prompt
    
    # 初期プロンプト生成
    base_prompt = generate_random_prompt()
    current_prompt = add_creative_randomness(base_prompt)
    PROMPT_HISTORY.append(current_prompt)
    print(f"🌱 初期プロンプト: {current_prompt}")
    
    # StreamDiffusion初期化
    stream = StreamDiffusion(prompt=current_prompt).stream
    
    # 入力ソース選択
    input_source = select_input_source()
    
    # フレームレシーバー初期化
    frame_receiver = None
    cap = None
    
    if input_source == "moon_frames":
        frame_receiver = FrameReceiver()
        if not frame_receiver.connect_to_sender():
            print("❌ main_moon.pyへの接続に失敗したため終了します")
            return
        if not frame_receiver.start_receiving():
            print("❌ フレーム受信開始に失敗したため終了します")
            return
        print("✨ main_moon.pyからのフレーム受信を開始しました")
    else:
        # カメラ選択
        camera_id = select_camera()
        if camera_id is None:
            print("❌ カメラが利用できないため終了します")
            return
        
        # カメラ初期化
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ カメラ {camera_id} を開けませんでした")
            return
    
    frame_count = 0
    creativity_update_interval = 30  # 30フレーム毎にクリエイティブ要素更新
    # save_interval = 100  # 自動保存を無効化
    
    # プロンプト入力スレッドを開始
    prompt_thread = threading.Thread(
        target=prompt_input_thread, 
        args=(stream, enhance_prompt),
        daemon=True
    )
    prompt_thread.start()
    
    # TCPサーバースレッドを開始
    server_thread = threading.Thread(
        target=run_server,
        args=(stream, enhance_prompt),
        daemon=True
    )
    server_thread.start()
    print(f"🌐 TCP サーバーを開始しました: {HOST}:{PORT}")
    
    print("\n===== StreamDiffusion Realtime UI =====")
    print("💡 ターミナルでいつでもプロンプトを入力可能")
    print(f"🌐 TCP接続でリモートからプロンプト変更: {HOST}:{PORT}")
    print("[r] ランダムプロンプトで再生成")
    print("[i] プロンプト入力モード")
    print("[s] 現在の画像を保存")
    print("[p] プロンプト履歴表示")
    print("[q] 終了")
    print("=======================================\n")
    
    while True:
        try:
            frame = None
            
            if input_source == "moon_frames":
                # main_moon.pyからフレーム取得
                moon_frame = frame_receiver.get_latest_frame()
                if moon_frame is not None:
                    frame = moon_frame
                else:
                    # フレームがまだない場合はスキップ
                    time.sleep(0.01)
                    continue
            else:
                # カメラからフレーム取得
                ret, frame = cap.read()
                if not ret:
                    print("カメラから映像が取得できませんでした")
                    break
            
            # カメラフレームをSDサイズにリサイズして初期画像に（任意）
            init_img = crop_center(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                   SD_SIDE_LENGTH * 2, SD_SIDE_LENGTH * 2).resize(
                (SD_SIDE_LENGTH, SD_SIDE_LENGTH), Image.NEAREST
            )
            
            # 推論用前処理
            image_tensor = stream.preprocess_image(init_img)
            
            try:
                # 推論実行
                output_image, fps = timeit(stream)(image=image_tensor)
                if isinstance(output_image, Image.Image):
                    # フレーム履歴管理（最大3フレーム）
                    if len(FRAME_HISTORY) >= MAX_FRAME_HISTORY:
                        FRAME_HISTORY.pop(0)  # 古いフレームを削除
                    
                    # 過去フレームとブレンド（3フレーム平均）
                    if FRAME_HISTORY:
                        curr_array = np.array(output_image, dtype=np.float32)
                        
                        # 過去フレームの平均を計算
                        history_arrays = [np.array(img, dtype=np.float32) for img in FRAME_HISTORY]
                        if history_arrays:
                            avg_history = np.mean(history_arrays, axis=0)
                            # 現在フレームと履歴の重み付き平均
                            blended = avg_history * (1 - FRAME_BLEND_ALPHA) + curr_array * FRAME_BLEND_ALPHA
                            output_image = Image.fromarray(blended.astype(np.uint8))
                    
                    FRAME_HISTORY.append(output_image)
                    
                    # 表示用にリサイズ
                    display_image = output_image.resize((DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.LANCZOS)
                    output_image_np = cv2.cvtColor(np.array(display_image), cv2.COLOR_RGB2BGR)
                    
                    # ウィンドウサイズを初回のみ設定
                    global window_initialized
                    if not window_initialized:
                        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
                        window_initialized = True
                    
                    cv2.imshow(WINDOW_NAME, output_image_np)

                    frame_count += 1
                    
                    # 定期的にクリエイティブ要素を更新
                    if frame_count % creativity_update_interval == 0:
                        creative_prompt = add_creative_randomness(current_prompt)
                        stream.prepare(
                            prompt=creative_prompt,
                            negative_prompt="low quality, bad quality, blurry, low resolution",
                            num_inference_steps=20,
                            guidance_scale=0.5,
                            delta=2.0,
                        )
                    
                    # 自動保存を無効化
                    # if frame_count % save_interval == 0:
                    #     save_to_gallery(output_image, current_prompt)
                    #     print(f"💾 自動保存（{frame_count}フレーム毎）")
            
            except Exception as e:
                print(f"❌ 生成中のエラー: {e}")
                if FRAME_HISTORY:
                    display_image = FRAME_HISTORY[-1].resize((DISPLAY_WIDTH, DISPLAY_HEIGHT), Image.LANCZOS)
                    fallback_np = cv2.cvtColor(np.array(display_image), cv2.COLOR_RGB2BGR)
                    cv2.imshow(WINDOW_NAME, fallback_np)

            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 終了します")
                break
            elif key == ord('r'):
                current_prompt = generate_random_prompt()
                PROMPT_HISTORY.append(current_prompt)
                print(f"🔁 新プロンプト: {current_prompt}")
                stream.prepare(
                    prompt=add_creative_randomness(current_prompt),
                    negative_prompt="low quality, bad quality, blurry, low resolution",
                    num_inference_steps=25,
                    guidance_scale=0.6,
                    delta=1.5,
                )
            elif key == ord('s') and FRAME_HISTORY:
                filename = save_to_gallery(FRAME_HISTORY[-1], current_prompt)
                print(f"💾 手動保存: {filename}")
            elif key == ord('i'):
                print("\n💡 プロンプト入力モード")
                print("追加オプション:")
                print("+ [テキスト] : 現在のプロンプトに追加")
                print("  [テキスト] : 新しいプロンプトに置換")
                print("  (空白)     : ランダムプロンプト")
                try:
                    user_input = input("プロンプトを入力: ").strip()
                    
                    if user_input == "":
                        new_prompt = generate_random_prompt()
                        print(f"🎲 ランダムプロンプト: {new_prompt}")
                    elif user_input.startswith("+"):
                        # 追加モード
                        addition = user_input[1:].strip()
                        if addition:
                            new_prompt = enhance_prompt(addition, current_prompt, append_mode=True)
                            print(f"➕ 追加されたプロンプト: {new_prompt}")
                        else:
                            print("❌ 追加する内容が空です")
                            continue
                    else:
                        # 通常入力は追加モード（累積）
                        new_prompt = enhance_prompt(user_input, current_prompt, append_mode=True)
                        print(f"➕ 追加されたプロンプト: {new_prompt}")
                    
                    current_prompt = new_prompt
                    PROMPT_HISTORY.append(new_prompt)
                    
                    stream.prepare(
                        prompt=add_creative_randomness(new_prompt),
                        negative_prompt="low quality, bad quality, blurry, low resolution", 
                        num_inference_steps=20,
                        guidance_scale=0.5,
                        delta=2.0,
                    )
                    print(f"🔄 プロンプトを更新しました: {new_prompt}")
                    
                except Exception as e:
                    print(f"❌ プロンプト更新エラー: {e}")
                    
            elif key == ord('p'):
                print("\n=== Prompt History ===")
                for i, prompt in enumerate(PROMPT_HISTORY, 1):
                    print(f"{i}. {prompt}")
                print("=======================\n")

        except KeyboardInterrupt:
            print("👋 キーボード割り込みによって終了")
            break

    # リソース解放
    if cap:
        cap.release()
    if frame_receiver:
        frame_receiver.stop_receiving()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
