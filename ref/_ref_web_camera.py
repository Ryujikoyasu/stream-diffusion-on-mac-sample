import time
import threading
import socket
import json
from typing import Any, Callable, List
from queue import Queue
import random
import sys, os

import cv2
import numpy as np
from PIL import Image, ImageChops
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

import speech_recognition as sr

from ..stream_diffusion import StreamDiffusion
from ..data.dictionaries import GOJUON_WORDS

SD_SIDE_LENGTH = 512

class PromptServer:
    def __init__(self, port=5000):
        self.port = port
        self.default_prompt = "moon, Cubism painting style, no humans, no people"
        self.current_prompt = self.default_prompt
        self.running = True
        self.prompt_queue = Queue()
        self.voice_recognition_active = True
        
        # 音韻処理用の変数を初期化
        self.last_used_phoneme = None
        
        # サーバーソケットの設定
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('localhost', self.port))
        self.server.listen(1)
        
        # サーバースレッドの開始
        self.thread = threading.Thread(target=self.run_server)
        self.thread.daemon = True
        self.thread.start()

        # 音声認識スレッドの開始
        self.voice_thread = threading.Thread(target=self.listen_for_voice_commands)
        self.voice_thread.daemon = True
        self.voice_thread.start()
        
        # 音声認識の初期化
        self.initialize_voice_recognition()
        
        # プロンプト管理の設定
        self.active_prompts = []          
        self.max_prompts = 6              # 最大プロンプト数を減らす
        self.prompt_update_interval = 30   # 更新間隔を30秒に延長
        self.last_prompt_update = time.time()
        self.stream_diffusion_lock = threading.Lock()
        
        print(f"Prompt server started on port {self.port}")
        print("Voice recognition is active. Say your prompt out loud.")

    def get_first_phoneme(self, text: str) -> str:
        """音声認識結果から最初の音韻を取得"""
        if not text:
            return ''
        return text[0]

    def get_random_words_from_phoneme(self, phoneme: str) -> list:
        """音韻から3つの単語と英語表現をランダムに選択"""
        if phoneme in GOJUON_WORDS:
            word_list = GOJUON_WORDS[phoneme]
            # リストから重複なしで3つをランダムに選択
            selected_pairs = random.sample(word_list, min(3, len(word_list)))
            return [pair[1] for pair in selected_pairs]  # 英語表現のリストを返す
        return []

    def process_voice_input(self, text: str):
        """音声入力を処理（改良版）"""
        try:
            if not text:
                return
            
            # テキストから最初の2文字分の音韻を取得
            first_two_phonemes = [char for char in text[:2] if char in GOJUON_WORDS]
            if not first_two_phonemes:
                return
            
            # 音韻の選択
            selected_phoneme = first_two_phonemes[0]  # デフォルトは最初の音韻
            
            # 最初の音韻が前回と同じ場合、2文字目があればそちらを使用
            if hasattr(self, 'last_used_phoneme') and selected_phoneme == self.last_used_phoneme and len(first_two_phonemes) > 1:
                selected_phoneme = first_two_phonemes[1]
            
            # 選択した音韻を記録
            self.last_used_phoneme = selected_phoneme
            
            # 単語の選択と処理
            english_words = self.get_random_words_from_phoneme(selected_phoneme)
            if english_words:
                print(f"音韻「{selected_phoneme}」から選択: → {', '.join(english_words)}")
                self.add_prompts(english_words)
                
        except Exception as e:
            print(f"音声処理中にエラーが発生しました: {e}")
            self.last_used_phoneme = None  # エラー時は前回の音韻をリセット

    def add_prompts(self, prompts: list):
        """複数のプロンプトを一度に追加（上限管理付き）"""
        # 重複を除去しながら新しいプロンプトを追加
        new_prompts = [p for p in prompts if p not in self.active_prompts]
        self.active_prompts.extend(new_prompts)
        
        # 上限を超えた場合、古いものから削除
        if len(self.active_prompts) > self.max_prompts:
            excess = len(self.active_prompts) - self.max_prompts
            self.active_prompts = self.active_prompts[excess:]
        
        # 更新されたプロンプトを生成
        combined_prompt = f"{self.default_prompt}, {', '.join(self.active_prompts)}"
        self.prompt_queue.put(combined_prompt)
        print(f"現在のプロンプト: {combined_prompt}")

    def listen_for_voice_commands(self):
        """音声認識処理（シンプル版）"""
        if not self.voice_recognition_active:
            print("音声認識は無効化されています")
            return

        while self.running:
            try:
                with sr.Microphone() as source:
                    if not hasattr(self, 'first_adjustment'):
                        print("マイクのノイズ調整を行います...")
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                        self.first_adjustment = True

                    print("音声認識待機中...")
                    try:
                        audio = self.recognizer.listen(source, phrase_time_limit=3)
                        text = self.recognizer.recognize_google(audio, language='ja-JP')
                        print(f"認識された音声: {text}")
                        
                        self.process_voice_input(text)

                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        print(f"音声認識サービスエラー: {e}")

            except Exception as e:
                print(f"予期せぬエラー: {e}")
                time.sleep(1)

    def run_server(self):
        """TCPサーバーを実行"""
        while self.running:
            try:
                client, addr = self.server.accept()
                print(f"Client connected from {addr}")
                
                while True:
                    data = client.recv(1024).decode('utf-8').strip()
                    if not data:
                        break
                    
                    print(f"Received prompt: {data}")
                    self.prompt_queue.put(data)
                    client.send(b"Prompt updated\n")
                
                client.close()
            except:
                pass

    def initialize_voice_recognition(self):
        """音声認識の初期化"""
        print("音声認識を初期化中...")
        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.energy_threshold = 4000
            print("音声認識の初期化が完了しました")
            
        except Exception as e:
            print(f"音声認識の初期化に失敗しました: {e}")
            print("音声認識を無効化します")
            self.voice_recognition_active = False

    def update_stream_diffusion(self, current_stream_diffusion):
        """StreamDiffusionインスタンスを更新"""
        current_time = time.time()
        
        # 前回の更新から30秒経過していない場合は更新しない
        if current_time - self.last_prompt_update < self.prompt_update_interval:
            # キューにプロンプトが溜まっていても、時間が経過していなければ更新しない
            while not self.prompt_queue.empty():
                _ = self.prompt_queue.get()  # キューをクリア
            return current_stream_diffusion, current_stream_diffusion.stream
        
        # 新しいプロンプトがキューにある場合は更新
        if not self.prompt_queue.empty():
            new_prompt = self.prompt_queue.get()
            # キューに残っているプロンプトをクリア
            while not self.prompt_queue.empty():
                _ = self.prompt_queue.get()
                
            if new_prompt != current_stream_diffusion.prompt:
                with self.stream_diffusion_lock:
                    try:
                        new_stream_diffusion = StreamDiffusion(prompt=new_prompt)
                        self.last_prompt_update = current_time
                        print(f"プロンプトを更新しました: {new_prompt} (次回更新まで {self.prompt_update_interval} 秒)")
                        return new_stream_diffusion, new_stream_diffusion.stream
                    except Exception as e:
                        print(f"新しいStreamDiffusionの作成に失敗: {e}")
        
        return current_stream_diffusion, current_stream_diffusion.stream

    def stop(self):
        """サーバーを停止"""
        self.running = False
        self.server.close()

def list_cameras():
    """利用可能なカメラの一覧を表示し、最後のインデックスを返す"""
    index = 0
    last_working_index = 0
    
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            print(f"Found camera at index: {index}")
            last_working_index = index
        cap.release()
        index += 1
    
    print(f"Using camera at index: {last_working_index}")
    return last_working_index

def blend_images(img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
    """2つの画像をブレンドする
    
    Args:
        img1: 1つ目の画像（前のフレーム）
        img2: 2つ目の画像（現在のフレーム）
        alpha: ブレンド率（0.0 〜 1.0）
    """
    return cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)

def main():
    try:
        camera_index = list_cameras()
        
        prompt_server = PromptServer()
        stream_diffusion = StreamDiffusion(prompt=prompt_server.current_prompt)
        stream = stream_diffusion.stream
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"カメラ {camera_index} を開けませんでした")
            
        # バッファサイズの最適化
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        img_dst = Image.new("RGB", (SD_SIDE_LENGTH, SD_SIDE_LENGTH))
        
        OUTPUT_WINDOW = "Stream Diffusion Output"
        
        # 初期状態でフルスクリーンウィンドウを作成
        cv2.namedWindow(OUTPUT_WINDOW, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(OUTPUT_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # 画面サイズを取得（フォールバック値として1920x1080を使用）
        try:
            screen = cv2.getWindowImageRect(OUTPUT_WINDOW)
            SCREEN_WIDTH = abs(screen[2]) or 1920
            SCREEN_HEIGHT = abs(screen[3]) or 1080
        except:
            SCREEN_WIDTH = 1920
            SCREEN_HEIGHT = 1080

        # 前のフレームを保存する変数を追加
        previous_output = None
        
        while True:
            _, frame = cap.read()

            # プロンプト更新の処理
            stream_diffusion, stream = prompt_server.update_stream_diffusion(stream_diffusion)
            
            init_img = crop_center(Image.fromarray(frame), SD_SIDE_LENGTH * 2, SD_SIDE_LENGTH * 2).resize(
                (SD_SIDE_LENGTH, SD_SIDE_LENGTH), Image.NEAREST
            )

            # テンソルをMPSデバイスに移動
            image_tensor = stream.preprocess_image(init_img).to(stream_diffusion.device)
            output_image, _ = timeit(stream)(image=image_tensor)

            if isinstance(output_image, Image.Image):
                img_dst = output_image
                output_array = np.array(img_dst)
                
                if previous_output is not None:
                    output_array = blend_images(previous_output, output_array, 0.6)
                
                previous_output = output_array.copy()

            # 以下は既存のディスプレイ処理
            background = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            
            # アスペクト比を維持しながら最大サイズを計算
            display_size = min(SCREEN_WIDTH, SCREEN_HEIGHT)
            
            # 画像をリサイズ
            resized_image = cv2.resize(output_array, (display_size, display_size), 
                                     interpolation=cv2.INTER_LANCZOS4)
            
            # 中央に配置するための座標を計算
            y_offset = (SCREEN_HEIGHT - display_size) // 2
            x_offset = (SCREEN_WIDTH - display_size) // 2
            
            # 画像を背景の中央に配置
            background[y_offset:y_offset+display_size, 
                     x_offset:x_offset+display_size] = resized_image
            
            cv2.imshow(OUTPUT_WINDOW, background)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESCで終了
                break

            # フレームレートを調整（より滑らかな表示のため）
            cv2.waitKey(33)  # 約30FPSに制限

    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        if 'prompt_server' in locals():
            prompt_server.stop()
        cv2.destroyAllWindows()


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

if __name__ == "__main__":
    main()
