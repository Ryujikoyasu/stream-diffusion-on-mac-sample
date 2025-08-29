import pygame
import pyaudio
import numpy as np
import math
import socket
import threading
import pickle
import time

# --- 設定項目 ---
# スクリーン設定
WIDTH, HEIGHT = 800, 800
FPS = 60

# オーディオ設定
CHUNK = 1024 * 2
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1

# TCP設定
FRAME_HOST = "127.0.0.1"
FRAME_PORT = 65433  # web_camera.pyとは別のポート

# 色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# --- オーディオリアクティブのパラメータ ---
BASS_RANGE = (60, 250)
MID_RANGE = (250, 2000)
HIGH_RANGE = (2000, 10000)
RIPPLE_THRESHOLD = 0.6  # 閾値を下げて波紋が起きやすく
last_bass_max = 0

# パーティクル設定
NUM_PARTICLES = 15

# 余韻効果用の値保持
audio_smoothing = {
    'bass': 0.0,
    'mid': 0.0,
    'high': 0.0,
    'volume': 0.0
}
SMOOTH_FACTOR = 0.85  # 余韻の強さ（0.9に近いほど長く残る）

# パーティクルクラス
class Particle:
    def __init__(self, x, y):
        self.center_x = x
        self.center_y = y
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.radius_from_center = np.random.uniform(100, 250)
        self.base_size = np.random.uniform(5, 20)
        self.speed = np.random.uniform(0.001, 0.005)
        self.color = (np.random.randint(150, 220), np.random.randint(150, 220), np.random.randint(150, 220))

    def update(self, mid_val, high_val):
        self.angle += self.speed * (1 + mid_val * 5)
        self.size = self.base_size * (1 + high_val * 3)
        self.x = self.center_x + self.radius_from_center * math.cos(self.angle)
        self.y = self.center_y + self.radius_from_center * math.sin(self.angle)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.size), 1)
        pygame.draw.circle(screen, self.color, (int(self.x+self.size*0.2), int(self.y-self.size*0.2)), int(self.size*0.7), 1)

# フレーム送信クラス（再接続対応）
class FrameSender:
    def __init__(self, host=FRAME_HOST, port=FRAME_PORT):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_conn = None
        self.running = False
        
    def start_server(self):
        """フレーム送信サーバーを開始"""
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        print(f"🖼️  フレーム送信サーバー開始: {self.host}:{self.port}")
        
    def _server_loop(self):
        """サーバーループ（再接続対応）"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            
            print(f"🔗 フレーム受信待機中: {self.host}:{self.port}")
            
            # 再接続ループ
            while self.running:
                try:
                    print("🔄 新しい接続を待機中...")
                    self.client_conn, addr = self.server_socket.accept()
                    print(f"✨ web_camera.py接続: {addr}")
                    
                    # 接続が有効な間はループ維持
                    self._handle_client_connection()
                    
                except Exception as e:
                    if self.running:
                        print(f"❌ 接続エラー: {e}")
                        self.client_conn = None
                        time.sleep(2)  # 少し待ってから再試行
        except Exception as e:
            print(f"❌ サーバー初期化エラー: {e}")
    
    def _handle_client_connection(self):
        """クライアント接続を処理（切断まで継続）"""
        while self.running and self.client_conn:
            try:
                # 接続が生きているかチェック
                self.client_conn.send(b'')  # キープアライブテスト
                time.sleep(0.1)
            except:
                print("🔌 接続が切断されました。再接続を待機します...")
                self.client_conn = None
                break
    
    def send_frame(self, frame_array):
        """フレームを送信（エラー時は接続をリセット）"""
        if not self.client_conn:
            return
            
        try:
            # numpy配列をpickleでシリアライズ
            frame_data = pickle.dumps(frame_array)
            frame_size = len(frame_data)
            
            # サイズを先に送信（4バイト）
            self.client_conn.sendall(frame_size.to_bytes(4, byteorder='big'))
            # フレームデータを送信
            self.client_conn.sendall(frame_data)
        except Exception as e:
            print(f"❌ フレーム送信エラー: {e}")
            print("🔄 接続をリセットして再接続を待機します")
            if self.client_conn:
                try:
                    self.client_conn.close()
                except:
                    pass
            self.client_conn = None
    
    def stop_server(self):
        """サーバー停止"""
        self.running = False
        if self.client_conn:
            try:
                self.client_conn.close()
            except:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

# 波紋クラス（余韻効果強化版）
class Ripple:
    def __init__(self, x, y, intensity=1.0):
        self.x = x
        self.y = y
        self.radius = 5
        self.max_radius = 600  # より大きく広がる
        self.speed = 2 * intensity  # インテンシティに応じて速度調整
        self.alpha = 180  # 初期透明度を下げて柔らか
        self.line_width = max(2, int(6 * intensity))
        self.fade_rate = 1.5  # ゆっくりフェード

    def update(self):
        self.radius += self.speed
        self.alpha = max(0, self.alpha - self.fade_rate)
        self.line_width = max(1, int(self.line_width * 0.995))  # より緩やかに細く
        # 速度も徐々に減速
        self.speed *= 0.998

    def draw(self, screen):
        if self.radius < self.max_radius and self.alpha > 0:
            s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            # 複数の円を重ねて柔らかい効果
            for i in range(3):
                offset_alpha = int(self.alpha * (0.7 - i * 0.2))
                offset_width = max(1, self.line_width - i)
                if offset_alpha > 0:
                    pygame.draw.circle(s, (WHITE[0], WHITE[1], WHITE[2], offset_alpha), 
                                     (self.x, self.y), int(self.radius + i), offset_width)
            screen.blit(s, (0, 0))

# --- メイン処理 ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Audio Reactive Visualizer - Frame Sender")
    clock = pygame.time.Clock()
    
    # フレーム送信サーバー開始
    frame_sender = FrameSender()
    frame_sender.start_server()

    try:
        moon_img = pygame.image.load("moon.png").convert_alpha()
        moon_img = pygame.transform.scale(moon_img, (100, 100))
        moon_rect = moon_img.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    except FileNotFoundError:
        print("moon.pngが見つかりません。画像なしで実行します。")
        moon_img = None

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    particles = [Particle(WIDTH // 2, HEIGHT // 2) for _ in range(NUM_PARTICLES)]
    ripples = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- オーディオデータの取得と解析 ---
        try:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            np_data = np.frombuffer(raw_data, dtype=np.int16)
            
            # --- <<< 修正ここから ---
            # データが空の場合、すべての値を0にしてこのフレームの処理をスキップ
            if np_data.size == 0:
                volume, bass_norm, mid_norm, high_norm = 0, 0, 0, 0
            else:
                # 整数オーバーフローを防ぐためにfloat型に変換
                np_data_float = np_data.astype(np.float32)
                
                # 1. 全体の音量 (RMS) - 感度を大幅に上げる
                volume = np.sqrt(np.mean(np_data_float**2)) / 300  # 1000 -> 300に変更
                
                # 2. 周波数スペクトル (FFT)
                fft_data = np.fft.rfft(np_data_float)
                fft_freq = np.fft.rfftfreq(len(np_data_float), 1.0/RATE)
                fft_amp = np.abs(fft_data) / CHUNK
                
                bass_val = np.mean(fft_amp[(fft_freq > BASS_RANGE[0]) & (fft_freq < BASS_RANGE[1])])
                mid_val = np.mean(fft_amp[(fft_freq > MID_RANGE[0]) & (fft_freq < MID_RANGE[1])])
                high_val = np.mean(fft_amp[(fft_freq > HIGH_RANGE[0]) & (fft_freq < HIGH_RANGE[1])])
                
                bass_norm = min(1.0, bass_val / 30.0)   # 100 -> 30に変更
                mid_norm = min(1.0, mid_val / 2.0)    # 5 -> 2に変更
                high_norm = min(1.0, high_val / 1.0)  # 2 -> 1に変更
                
                # 余韻効果を適用（スムージング）
                global audio_smoothing
                audio_smoothing['bass'] = audio_smoothing['bass'] * SMOOTH_FACTOR + bass_norm * (1 - SMOOTH_FACTOR)
                audio_smoothing['mid'] = audio_smoothing['mid'] * SMOOTH_FACTOR + mid_norm * (1 - SMOOTH_FACTOR)
                audio_smoothing['high'] = audio_smoothing['high'] * SMOOTH_FACTOR + high_norm * (1 - SMOOTH_FACTOR)
                audio_smoothing['volume'] = audio_smoothing['volume'] * SMOOTH_FACTOR + volume * (1 - SMOOTH_FACTOR)
                
                # スムージングした値を使用
                bass_norm = audio_smoothing['bass']
                mid_norm = audio_smoothing['mid'] 
                high_norm = audio_smoothing['high']
                volume = audio_smoothing['volume']
            # --- 修正ここまで >>> ---

        except (IOError, ValueError):
            volume, bass_norm, mid_norm, high_norm = 0, 0, 0, 0

        # --- 描画処理 ---
        screen.fill(BLACK)

        global last_bass_max
        # 波紋発生条件を緩和し、強度に応じて複数生成
        if bass_norm > RIPPLE_THRESHOLD:
            if bass_norm > last_bass_max * 1.3:  # 1.5 -> 1.3に緩和
                # 強いビートで複数の波紋
                if bass_norm > 0.8:
                    ripples.append(Ripple(WIDTH // 2, HEIGHT // 2, bass_norm))
                    # 少しずらして追加の波紋
                    ripples.append(Ripple(WIDTH // 2 + 20, HEIGHT // 2 + 20, bass_norm * 0.7))
                else:
                    ripples.append(Ripple(WIDTH // 2, HEIGHT // 2, bass_norm))
        last_bass_max = bass_norm * 0.95  # 減衰を緩やか
        
        for ripple in ripples[:]:
            ripple.update()
            if ripple.alpha <= 0:
                ripples.remove(ripple)
            else:
                ripple.draw(screen)

        for particle in particles:
            particle.update(mid_norm, high_norm)
            particle.draw(screen)

        center = (WIDTH // 2, HEIGHT // 2)
        
        # --- <<< 修正: volumeがNaNでないことを確認 ---
        # 月を最初に描画（背景として）
        if moon_img:
            screen.blit(moon_img, moon_rect)
        else:
            pygame.draw.circle(screen, WHITE, center, 50)
        
        # 音響エフェクトを月の周りに描画（月の部分を除外）
        if not np.isnan(volume):
            glow_radius = int(60 + volume * 8)  # 感度を大幅アップ（3 -> 8）
            glow_alpha = min(200, int(50 + volume * 10))  # アルファも強化（3 -> 10）
            if glow_radius > 60:
                # グロー効果用のサーフェス
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                
                # 多層のグローで柔らかい効果
                for i in range(4):  # 層を増やして更に柔らかく
                    layer_radius = glow_radius - i * 8
                    layer_alpha = glow_alpha // (i + 1)
                    if layer_radius > 0 and layer_alpha > 0:
                        # 色も音の強さに応じて変化
                        color_intensity = min(255, int(200 + volume * 2))
                        
                        # 月の範囲を除外するためのマスクを作成
                        mask_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                        
                        # 外側の円（グロー）
                        pygame.draw.circle(mask_surface, (color_intensity, color_intensity, 150, layer_alpha), 
                                         (glow_radius, glow_radius), layer_radius)
                        
                        # 月の部分（中心から50ピクセル）を黒で塗りつぶして除外
                        pygame.draw.circle(mask_surface, (0, 0, 0, 0), 
                                         (glow_radius, glow_radius), 55)  # 月より少し大きめ
                        
                        s.blit(mask_surface, (0, 0))
                
                screen.blit(s, (center[0] - glow_radius, center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.display.flip()
        
        # Pygameサーフェスをnumpy配列として取得
        frame_array = pygame.surfarray.array3d(screen)
        # Pygameの座標系(x,y,rgb) -> OpenCVの座標系(y,x,bgr)に変換
        frame_array = np.transpose(frame_array, (1, 0, 2))
        # RGB -> BGR変換
        frame_array = frame_array[:, :, ::-1]
        
        # フレームを送信
        frame_sender.send_frame(frame_array)
        
        clock.tick(FPS)

    stream.stop_stream()
    stream.close()
    p.terminate()
    frame_sender.stop_server()
    pygame.quit()

if __name__ == '__main__':
    main()