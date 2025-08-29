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

# 余韻効果用の値保持
audio_smoothing = {
    'bass': 0.0,
    'mid': 0.0,
    'high': 0.0,
    'volume': 0.0
}
SMOOTH_FACTOR = 0.88  # 曼荼羅用にさらに長い余韻

# 美しい波の曼荼羅設定
NUM_WAVE_POINTS = 360  # 滑らかな波のための点数
NUM_WAVE_LAYERS = 8    # 波の層数
WAVE_FREQUENCIES = [2, 3, 5, 8, 13, 21]  # フィボナッチ数列による波の周波数

# 波の層クラス
class WaveLayer:
    def __init__(self, center_x, center_y, base_radius, frequency, phase_offset=0):
        self.center_x = center_x
        self.center_y = center_y
        self.base_radius = base_radius
        self.frequency = frequency
        self.phase = phase_offset
        self.amplitude = 0
        self.color_phase = np.random.uniform(0, 2 * np.pi)
        
    def update(self, bass_val, mid_val, high_val, time_factor):
        # 位相を時間と音に応じて更新
        self.phase += 0.02 + mid_val * 0.05
        self.color_phase += 0.01 + high_val * 0.02
        
        # 振幅を音に応じて変化（より滑らか）
        self.amplitude = 15 + bass_val * 40 + high_val * 20
        
    def get_wave_points(self):
        """波の座標点を生成"""
        points = []
        for i in range(NUM_WAVE_POINTS):
            angle = (i * 2 * np.pi) / NUM_WAVE_POINTS
            
            # 複数周波数の波を重ね合わせ
            wave_offset = 0
            for freq in WAVE_FREQUENCIES[:3]:  # 3つの周波数を重ね合わせ
                wave_offset += math.sin(angle * freq + self.phase) * (self.amplitude / len(WAVE_FREQUENCIES))
            
            # 動的半径
            dynamic_radius = self.base_radius + wave_offset
            
            x = self.center_x + dynamic_radius * math.cos(angle)
            y = self.center_y + dynamic_radius * math.sin(angle)
            points.append((int(x), int(y)))
            
        return points
    
    def get_color(self, volume):
        """美しいグラデーション色を生成"""
        # HSV -> RGB変換（滑らかな色変化）
        hue = (self.color_phase + self.frequency * 0.3) % (2 * np.pi)
        saturation = min(1.0, 0.7 + volume * 0.3)
        value = min(1.0, 0.6 + volume * 0.4)
        
        # 簡易HSV->RGB変換
        c = value * saturation
        x = c * (1 - abs(((hue * 3 / np.pi) % 2) - 1))
        m = value - c
        
        if hue < np.pi / 3:
            r, g, b = c, x, 0
        elif hue < 2 * np.pi / 3:
            r, g, b = x, c, 0
        elif hue < np.pi:
            r, g, b = 0, c, x
        elif hue < 4 * np.pi / 3:
            r, g, b = 0, x, c
        elif hue < 5 * np.pi / 3:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        # 色値を0-255の範囲に正規化してクランプ
        r_final = max(0, min(255, int((r + m) * 255)))
        g_final = max(0, min(255, int((g + m) * 255)))
        b_final = max(0, min(255, int((b + m) * 255)))
        
        return (r_final, g_final, b_final)

# 光のガラス玉クラス
class LightOrb:
    def __init__(self, x, y, base_size, color_phase=0):
        self.x = x
        self.y = y
        self.base_size = base_size
        self.color_phase = color_phase
        self.glow_intensity = 0
        self.size_multiplier = 1.0
        self.position_offset_x = 0
        self.position_offset_y = 0
        self.time = 0
        
    def update(self, bass_val, mid_val, high_val, time_factor):
        self.time = time_factor
        
        # BASSでサイズが変化
        self.size_multiplier = 1.0 + bass_val * 0.6
        
        # MIDで位置が微細に変化（浮遊効果）
        self.position_offset_x = mid_val * 15 * math.sin(time_factor * 1.5 + self.color_phase)
        self.position_offset_y = mid_val * 12 * math.cos(time_factor * 1.2 + self.color_phase * 1.3)
        
        # HIGHで光の強度が変化
        self.glow_intensity = high_val * 0.8
        
    def draw(self, screen, volume):
        """光のガラス玉を描画"""
        temp_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        # 現在の位置とサイズ
        current_x = int(self.x + self.position_offset_x)
        current_y = int(self.y + self.position_offset_y)
        current_size = int(self.base_size * self.size_multiplier)
        
        if current_size > 0:
            # 1. 外側のグロー効果
            for glow_layer in range(8):
                glow_radius = current_size + glow_layer * 4
                glow_alpha = max(3, int(15 * (1 + self.glow_intensity) - glow_layer * 2))
                
                # グローの色（虹色効果）
                glow_hue = (self.color_phase + self.time * 0.5 + glow_layer * 0.2) % (2 * np.pi)
                glow_brightness = 0.4 + self.glow_intensity * 0.6
                
                r = max(0, min(255, int(150 + 105 * math.sin(glow_hue) * glow_brightness)))
                g = max(0, min(255, int(150 + 105 * math.sin(glow_hue + 2*np.pi/3) * glow_brightness)))
                b = max(0, min(255, int(150 + 105 * math.sin(glow_hue + 4*np.pi/3) * glow_brightness)))
                
                try:
                    pygame.draw.circle(temp_surface, (r, g, b, glow_alpha), 
                                     (current_x, current_y), glow_radius, 1)
                except (ValueError, TypeError):
                    pass
            
            # 2. メインのガラス玉本体
            main_alpha = max(30, int(80 + volume * 100))
            main_hue = (self.color_phase + self.time * 0.3) % (2 * np.pi)
            main_brightness = 0.7 + self.glow_intensity * 0.3
            
            r = max(0, min(255, int(180 + 75 * math.sin(main_hue) * main_brightness)))
            g = max(0, min(255, int(180 + 75 * math.sin(main_hue + 2*np.pi/3) * main_brightness)))
            b = max(0, min(255, int(180 + 75 * math.sin(main_hue + 4*np.pi/3) * main_brightness)))
            
            try:
                pygame.draw.circle(temp_surface, (r, g, b, main_alpha), 
                                 (current_x, current_y), current_size, 2)
            except (ValueError, TypeError):
                pass
                
            # 3. ハイライト（光の反射）
            highlight_size = max(3, current_size // 3)
            highlight_x = current_x - current_size // 4
            highlight_y = current_y - current_size // 4
            highlight_alpha = max(40, int(120 + self.glow_intensity * 135))
            
            try:
                pygame.draw.circle(temp_surface, (255, 255, 255, highlight_alpha), 
                                 (highlight_x, highlight_y), highlight_size)
            except (ValueError, TypeError):
                pass
        
        screen.blit(temp_surface, (0, 0))

# 本格的な曼荼羅システム
class WaveMandala:
    def __init__(self, center_x, center_y):
        self.center_x = center_x
        self.center_y = center_y
        self.wave_layers = []
        self.time = 0
        self.symmetry_points = 8  # 8方向対称
        
        # 複数の波の層を作成
        for i in range(NUM_WAVE_LAYERS):
            radius = 60 + i * 35
            frequency = WAVE_FREQUENCIES[i % len(WAVE_FREQUENCIES)]
            phase_offset = i * np.pi / 4
            layer = WaveLayer(center_x, center_y, radius, frequency, phase_offset)
            self.wave_layers.append(layer)
        
        # 対称配置のサブセンター
        self.sub_centers = []
        for i in range(self.symmetry_points):
            angle = (i * 2 * np.pi) / self.symmetry_points
            sub_x = center_x + 150 * math.cos(angle)
            sub_y = center_y + 150 * math.sin(angle)
            self.sub_centers.append((sub_x, sub_y))
        
        # 光のガラス玉を配置
        self.light_orbs = []
        
        # 中心の大きなガラス玉
        self.light_orbs.append(LightOrb(center_x, center_y, 25, 0))
        
        # 対称配置の中サイズガラス玉
        for i in range(8):
            angle = (i * 2 * np.pi) / 8
            orb_x = center_x + 120 * math.cos(angle)
            orb_y = center_y + 120 * math.sin(angle)
            self.light_orbs.append(LightOrb(orb_x, orb_y, 18, i * np.pi / 4))
        
        # 外周の小さなガラス玉
        for i in range(16):
            angle = (i * 2 * np.pi) / 16
            orb_x = center_x + 250 * math.cos(angle)
            orb_y = center_y + 250 * math.sin(angle)
            self.light_orbs.append(LightOrb(orb_x, orb_y, 12, i * np.pi / 8))
        
        # ランダム配置の小さなガラス玉
        for i in range(12):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(80, 200)
            orb_x = center_x + distance * math.cos(angle)
            orb_y = center_y + distance * math.sin(angle)
            self.light_orbs.append(LightOrb(orb_x, orb_y, 8, np.random.uniform(0, 2 * np.pi)))
    
    def update(self, bass_val, mid_val, high_val):
        self.time += 0.016  # 60FPSでの時間増分
        
        for layer in self.wave_layers:
            layer.update(bass_val, mid_val, high_val, self.time)
        
        # 光のガラス玉を更新
        for orb in self.light_orbs:
            orb.update(bass_val, mid_val, high_val, self.time)
    
    def draw(self, screen, bass_val, mid_val, high_val, volume):
        # 感度を適度に調整（1.5倍程度）
        bass_enhanced = bass_val * 1.5
        mid_enhanced = mid_val * 1.5  
        high_enhanced = high_val * 1.5
        
        # 1. 多層波紋パターン（BASS反応）
        self._draw_wave_ripples(screen, bass_enhanced, mid_enhanced, high_enhanced, volume)
        
        # 2. 波状放射パターン（MID反応） 
        self._draw_wavy_radials(screen, bass_enhanced, mid_enhanced, high_enhanced, volume)
        
        # 3. 流動的な対称波パターン
        self._draw_flowing_symmetric_waves(screen, bass_enhanced, mid_enhanced, high_enhanced, volume)
        
        # 4. 波の干渉パターン
        self._draw_wave_interference(screen, bass_enhanced, mid_enhanced, high_enhanced, volume)
        
        # 5. ぼやけた中心波動
        self._draw_blurred_center_waves(screen, bass_enhanced, mid_enhanced, high_enhanced, volume)
        
        # 6. ぼやけた中心コア
        self._draw_blurred_center_core(screen, bass_enhanced, mid_enhanced, high_enhanced, volume)
        
        # 7. 光のガラス玉を描画（最前面）
        for orb in self.light_orbs:
            orb.draw(screen, volume)
    
    def _draw_wave_at_point(self, screen, layer, center, color, alpha, volume_multiplier):
        """指定した点で波を描画"""
        try:
            # 一時的にレイヤーの中心を変更
            original_x, original_y = layer.center_x, layer.center_y
            layer.center_x, layer.center_y = int(center[0]), int(center[1])
            
            points = layer.get_wave_points()
            if len(points) > 2:
                closed_points = points + [points[0]]
                
                for width_idx, width in enumerate([3, 2, 1]):
                    layer_alpha = max(10, int(alpha * volume_multiplier // (width_idx + 1)))
                    
                    temp_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                    
                    try:
                        draw_color = (*color, layer_alpha)
                        pygame.draw.lines(temp_surface, draw_color, False, closed_points, width)
                    except (ValueError, TypeError):
                        pygame.draw.circle(temp_surface, color, center, layer.base_radius // 3, width)
                    
                    screen.blit(temp_surface, (0, 0))
            
            # 元の中心に戻す
            layer.center_x, layer.center_y = original_x, original_y
        except Exception:
            pass
    
    def _draw_wave_ripples(self, screen, bass_val, mid_val, high_val, volume):
        """多層波紋パターン（BASSで波紋の強度変化）"""
        temp_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        # 複数の波紋層を重ねて複雑性を作る
        for ripple_set in range(4):
            for i in range(12):
                # BASSで波紋の広がりが変化（適度な感度）
                bass_effect = 1.0 + bass_val * 0.8
                base_radius = 30 + i * 30 + ripple_set * 15
                radius = int(base_radius * bass_effect)
                
                # 波の変調でぼやけた効果
                wave_modulation = 5 + mid_val * 10
                radius += int(wave_modulation * math.sin(self.time * 2 + i * 0.3 + ripple_set * 0.8))
                
                if radius > 0:
                    # ぼやけた透明度で層を重ねる
                    alpha = max(5, int(20 + volume * 40 - i * 2 - ripple_set * 3))
                    
                    # HIGHで色の変化（適度な感度）
                    hue = (self.time * 0.5 + i * 0.4 + ripple_set * 0.6) % (2 * np.pi)
                    color_shift = high_val * 0.7
                    
                    r = max(0, min(255, int(100 + 100 * math.sin(hue + color_shift))))
                    g = max(0, min(255, int(100 + 100 * math.sin(hue + color_shift + 2*np.pi/3))))
                    b = max(0, min(255, int(100 + 100 * math.sin(hue + color_shift + 4*np.pi/3))))
                    
                    try:
                        # 細い線でぼやけた効果
                        width = 1 if ripple_set > 1 else 2
                        pygame.draw.circle(temp_surface, (r, g, b, alpha), 
                                         (self.center_x, self.center_y), radius, width)
                    except (ValueError, TypeError):
                        pass
        
        screen.blit(temp_surface, (0, 0))
    
    def _draw_wavy_radials(self, screen, bass_val, mid_val, high_val, volume):
        """波状放射パターン（MID反応 - ぼやけた放射波）"""
        temp_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        # 複数層の波状放射線
        for layer_set in range(3):
            radial_count = 24 + layer_set * 8  # 24, 32, 40本
            
            for i in range(radial_count):
                base_angle = (i * 2 * np.pi) / radial_count
                
                # MIDで回転が変化（適度な感度）
                mid_rotation = mid_val * 1.2 + self.time * (0.8 + layer_set * 0.3)
                angle = base_angle + mid_rotation
                
                # 波状の長さ変化でぼやけた効果
                base_length = 80 + layer_set * 40
                wave_length = base_length + mid_val * 80  # MIDで伸縮
                
                # 波の変調を加えてぼやけた線に
                wave_steps = 20
                points = []
                
                for step in range(wave_steps):
                    t = step / wave_steps
                    current_length = t * wave_length
                    
                    # 波の歪みでぼやけ効果
                    wave_distortion = 3 + bass_val * 8
                    distorted_angle = angle + wave_distortion * math.sin(t * 4 + self.time * 2) * 0.1
                    
                    x = self.center_x + current_length * math.cos(distorted_angle)
                    y = self.center_y + current_length * math.sin(distorted_angle)
                    
                    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                        points.append((int(x), int(y)))
                
                # ぼやけた色（HIGHで変化）
                alpha = max(8, int(25 + volume * 50 - layer_set * 8))
                hue_phase = i * 0.15 + layer_set * 0.5 + self.time * 0.3
                color_intensity = 0.6 + high_val * 0.5
                
                r = max(0, min(255, int(80 + 120 * math.sin(hue_phase) * color_intensity)))
                g = max(0, min(255, int(80 + 120 * math.sin(hue_phase + 2*np.pi/3) * color_intensity)))
                b = max(0, min(255, int(80 + 120 * math.sin(hue_phase + 4*np.pi/3) * color_intensity)))
                
                try:
                    if len(points) > 1:
                        width = 1 if layer_set > 0 else 2
                        pygame.draw.lines(temp_surface, (r, g, b, alpha), False, points, width)
                except (ValueError, TypeError):
                    pass
        
        screen.blit(temp_surface, (0, 0))
    
    def _draw_flowing_symmetric_waves(self, screen, bass_val, mid_val, high_val, volume):
        """流動的な対称波パターン（ぼやけた曼荼羅感）"""
        temp_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        # 8方向の対称波（複雑性を保ちつつ軽量）
        for symmetry_id in range(8):
            base_angle = (symmetry_id * 2 * np.pi) / 8
            
            # 複数の波形を重ねる
            for wave_freq in [2, 3, 5]:  # フィボナッチ周波数
                points = []
                wave_amplitude = 15 + bass_val * 25  # BASSで振幅変化
                wave_length = 150 + mid_val * 100   # MIDで長さ変化
                
                # 波形を生成
                for i in range(40):  # 適度な精度
                    t = i / 40
                    distance = t * wave_length
                    
                    # 複数周波数の波を重ね合わせ
                    wave_offset = wave_amplitude * math.sin(t * wave_freq * 2 * np.pi + self.time * 2 + symmetry_id * 0.5)
                    angle = base_angle + wave_offset * 0.02  # 角度にも波の変調
                    
                    x = self.center_x + distance * math.cos(angle)
                    y = self.center_y + distance * math.sin(angle)
                    
                    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                        points.append((int(x), int(y)))
                
                # ぼやけた透明色
                alpha = max(10, int(30 + volume * 60 - wave_freq * 8))
                hue_shift = symmetry_id * 0.3 + wave_freq * 0.4 + self.time * 0.4
                color_intensity = 0.7 + high_val * 0.4  # HIGHで色変化
                
                r = max(0, min(255, int(90 + 130 * math.sin(hue_shift) * color_intensity)))
                g = max(0, min(255, int(90 + 130 * math.sin(hue_shift + 2*np.pi/3) * color_intensity)))
                b = max(0, min(255, int(90 + 130 * math.sin(hue_shift + 4*np.pi/3) * color_intensity)))
                
                try:
                    if len(points) > 1:
                        pygame.draw.lines(temp_surface, (r, g, b, alpha), False, points, 1)
                except (ValueError, TypeError):
                    pass
        
        screen.blit(temp_surface, (0, 0))
    
    def _draw_wave_interference(self, screen, bass_val, mid_val, high_val, volume):
        """波の干渉パターン（複雑な曼荼羅効果）"""
        temp_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        # 複数の波源による干渉
        wave_sources = [(self.center_x, self.center_y)]
        # 対称位置に波源を追加（複雑性）
        for i in range(6):
            angle = (i * 2 * np.pi) / 6
            source_distance = 60 + bass_val * 40
            source_x = self.center_x + source_distance * math.cos(angle + self.time * 0.5)
            source_y = self.center_y + source_distance * math.sin(angle + self.time * 0.5)
            wave_sources.append((source_x, source_y))
        
        # 干渉波の描画（ぼやけた同心円群）
        for source_id, (src_x, src_y) in enumerate(wave_sources):
            for ring in range(8):
                # 波の半径（音響反応）
                base_radius = 20 + ring * 25
                wave_modulation = bass_val * 15 + mid_val * 10
                radius = int(base_radius + wave_modulation * math.sin(self.time * 3 + source_id * 0.8 + ring * 0.4))
                
                if radius > 0:
                    # 重なりによる複雑な透明度
                    alpha = max(5, int(15 + volume * 30 - ring * 2))
                    
                    # HIGHで色相変化
                    hue = (self.time * 0.4 + source_id * 0.6 + ring * 0.3 + high_val * 0.8) % (2 * np.pi)
                    
                    r = max(0, min(255, int(70 + 120 * math.sin(hue))))
                    g = max(0, min(255, int(70 + 120 * math.sin(hue + 2*np.pi/3))))
                    b = max(0, min(255, int(70 + 120 * math.sin(hue + 4*np.pi/3))))
                    
                    try:
                        pygame.draw.circle(temp_surface, (r, g, b, alpha), 
                                         (int(src_x), int(src_y)), radius, 1)
                    except (ValueError, TypeError):
                        pass
        
        screen.blit(temp_surface, (0, 0))
    
    def _draw_blurred_center_waves(self, screen, bass_val, mid_val, high_val, volume):
        """ぼやけた中心波動（柔らかい曼荼羅コア）"""
        temp_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        # 複数のぼやけた波形パターン
        for wave_set in range(6):
            wave_radius = 20 + wave_set * 12
            point_count = 36  # 滑らかな円形
            points = []
            
            for i in range(point_count):
                angle = (i * 2 * np.pi) / point_count
                
                # 複数の周波数を重ね合わせた波
                wave_1 = math.sin(angle * 3 + self.time * 2 + wave_set * 0.5)
                wave_2 = math.sin(angle * 5 + self.time * 1.5 + bass_val * 2)
                wave_3 = math.sin(angle * 8 + self.time * 3 + mid_val * 3)
                
                # 波の合成でぼやけた輪郭
                wave_distortion = (wave_1 + wave_2 * 0.7 + wave_3 * 0.5) / 3
                distorted_radius = wave_radius + wave_distortion * (8 + bass_val * 15)
                
                x = self.center_x + distorted_radius * math.cos(angle)
                y = self.center_y + distorted_radius * math.sin(angle)
                points.append((int(x), int(y)))
            
            # ぼやけた色とグラデーション
            alpha = max(15, int(40 + volume * 80 - wave_set * 6))
            hue_base = wave_set * 0.4 + self.time * 0.3
            
            # HIGHで色の鮮やかさ変化
            color_boost = 0.8 + high_val * 0.6
            r = max(0, min(255, int(120 + 100 * math.sin(hue_base + bass_val) * color_boost)))
            g = max(0, min(255, int(120 + 100 * math.sin(hue_base + mid_val + 2*np.pi/3) * color_boost)))
            b = max(0, min(255, int(120 + 100 * math.sin(hue_base + high_val + 4*np.pi/3) * color_boost)))
            
            try:
                if len(points) > 2:
                    # 閉じた波形でぼやけ効果
                    closed_points = points + [points[0]]
                    width = 1 if wave_set > 2 else 2
                    pygame.draw.lines(temp_surface, (r, g, b, alpha), False, closed_points, width)
            except (ValueError, TypeError):
                pass
        
        screen.blit(temp_surface, (0, 0))
    
    def _draw_blurred_center_core(self, screen, bass_val, mid_val, high_val, volume):
        """ぼやけた中心コア（曼荼羅的な波の中心）"""
        temp_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        # 重層的なぼやけた波形パターン
        for core_layer in range(12):  # 複雑性を戻す
            # 各層で異なる波のパターン
            frequency = 2 + core_layer * 0.5
            base_radius = 8 + core_layer * 6
            point_count = 48  # 滑らかな波形
            points = []
            
            for i in range(point_count):
                angle = (i * 2 * np.pi) / point_count
                
                # 多重波の合成でぼやけた形
                wave_1 = math.sin(angle * frequency + self.time * 2 + core_layer * 0.3)
                wave_2 = math.sin(angle * (frequency + 1) + self.time * 1.5 + bass_val * 1.5)
                wave_3 = math.sin(angle * (frequency * 2) + self.time * 3 + mid_val * 2)
                
                # 波の重ね合わせ（適度な感度）
                wave_amplitude = 3 + bass_val * 8 + core_layer * 0.8
                wave_sum = (wave_1 + wave_2 * 0.6 + wave_3 * 0.4) / 3
                distorted_radius = base_radius + wave_amplitude * wave_sum
                
                x = self.center_x + distorted_radius * math.cos(angle)
                y = self.center_y + distorted_radius * math.sin(angle)
                points.append((int(x), int(y)))
            
            # 重層的な透明度でぼやけ効果
            alpha = max(8, int(25 + volume * 50 - core_layer * 3))
            
            # HIGHで色相がゆっくり変化
            hue_base = core_layer * 0.3 + self.time * 0.2 + high_val * 0.6
            color_intensity = 0.7 + high_val * 0.4
            
            r = max(0, min(255, int(100 + 120 * math.sin(hue_base) * color_intensity)))
            g = max(0, min(255, int(100 + 120 * math.sin(hue_base + 2*np.pi/3) * color_intensity)))
            b = max(0, min(255, int(100 + 120 * math.sin(hue_base + 4*np.pi/3) * color_intensity)))
            
            try:
                if len(points) > 2:
                    closed_points = points + [points[0]]
                    pygame.draw.lines(temp_surface, (r, g, b, alpha), False, closed_points, 1)
            except (ValueError, TypeError):
                pass
        
        screen.blit(temp_surface, (0, 0))

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

# --- メイン処理 ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Audio Reactive Mandala - Frame Sender")
    clock = pygame.time.Clock()
    
    # フレーム送信サーバー開始
    frame_sender = FrameSender()
    frame_sender.start_server()

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    # 波の曼荼羅システム初期化
    mandala = WaveMandala(WIDTH // 2, HEIGHT // 2)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- オーディオデータの取得と解析 ---
        try:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            np_data = np.frombuffer(raw_data, dtype=np.int16)
            
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

        except (IOError, ValueError):
            volume, bass_norm, mid_norm, high_norm = 0, 0, 0, 0

        # --- 描画処理 ---
        screen.fill(BLACK)

        # 曼荼羅システム更新・描画
        mandala.update(bass_norm, mid_norm, high_norm)
        mandala.draw(screen, bass_norm, mid_norm, high_norm, volume)

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