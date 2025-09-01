import json
import os
from pathlib import Path
from typing import Dict, Any, List

class Config:
    """Configuration manager for stream-diffusion-on-mac-sample"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get(self, key_path: str, default=None):
        """Get configuration value by dot notation (e.g., 'display.width')"""
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    # Display settings
    @property
    def width(self) -> int:
        return self.get('display.width', 800)
    
    @property
    def height(self) -> int:
        return self.get('display.height', 800)
    
    @property
    def fps(self) -> int:
        return self.get('display.fps', 60)
    
    @property
    def display_width(self) -> int:
        return self.get('display.display_width', 2048)
    
    @property
    def display_height(self) -> int:
        return self.get('display.display_height', 2048)
    
    # Audio settings
    @property
    def audio_chunk(self) -> int:
        return self.get('audio.chunk', 2048)
    
    @property
    def audio_rate(self) -> int:
        return self.get('audio.rate', 44100)
    
    @property
    def audio_channels(self) -> int:
        return self.get('audio.channels', 1)
    
    @property
    def bass_range(self) -> List[int]:
        return self.get('audio.bass_range', [60, 250])
    
    @property
    def mid_range(self) -> List[int]:
        return self.get('audio.mid_range', [250, 2000])
    
    @property
    def high_range(self) -> List[int]:
        return self.get('audio.high_range', [2000, 10000])
    
    @property
    def smooth_factor(self) -> float:
        return self.get('audio.smooth_factor', 0.85)
    
    # Network settings
    @property
    def host(self) -> str:
        return self.get('network.host', '127.0.0.1')
    
    @property
    def frame_port(self) -> int:
        return self.get('network.frame_port', 65433)
    
    @property
    def tcp_port(self) -> int:
        return self.get('network.tcp_port', 65432)
    
    # StreamDiffusion settings
    @property
    def sd_side_length(self) -> int:
        return self.get('streamdiffusion.sd_side_length', 512)
    
    @property
    def model_id(self) -> str:
        return self.get('streamdiffusion.model_id', 'stabilityai/sd-turbo')
    
    @property
    def guidance_scale(self) -> float:
        return self.get('streamdiffusion.guidance_scale', 0.6)
    
    @property
    def delta(self) -> float:
        return self.get('streamdiffusion.delta', 1.5)
    
    @property
    def num_inference_steps(self) -> int:
        return self.get('streamdiffusion.num_inference_steps', 25)
    
    @property
    def use_random_seed(self) -> bool:
        return self.get('streamdiffusion.use_random_seed', True)
    
    @property
    def optimize_for_speed(self) -> bool:
        return self.get('streamdiffusion.optimize_for_speed', True)
    
    @property
    def use_kohaku_model(self) -> bool:
        return self.get('streamdiffusion.use_kohaku_model', False)
    
    # Creativity settings
    @property
    def frame_blend_alpha(self) -> float:
        return self.get('creativity.frame_blend_alpha', 0.3)
    
    @property
    def max_frame_history(self) -> int:
        return self.get('creativity.max_frame_history', 3)
    
    @property
    def creativity_update_interval(self) -> int:
        return self.get('creativity.creativity_update_interval', 30)
    
    @property
    def themes(self) -> List[str]:
        return self.get('creativity.themes', [
            "nature and wildlife", "cyberpunk city", "fantasy creatures",
            "space exploration", "underwater world", "steampunk invention",
            "magical forest", "futuristic technology"
        ])
    
    @property
    def creative_modifiers(self) -> List[str]:
        return self.get('creativity.creative_modifiers', [
            "vibrant colors", "soft pastels", "neon glow", "watercolor style", 
            "oil painting", "abstract expressionism", "surreal", "dreamy atmosphere"
        ])
    
    # Mandala settings
    @property
    def num_wave_points(self) -> int:
        return self.get('mandala.num_wave_points', 360)
    
    @property
    def num_wave_layers(self) -> int:
        return self.get('mandala.num_wave_layers', 8)
    
    @property
    def wave_frequencies(self) -> List[int]:
        return self.get('mandala.wave_frequencies', [2, 3, 5, 8, 13, 21])
    
    # Path settings
    @property
    def gallery_dir(self) -> str:
        return self.get('paths.gallery_dir', 'gallery')
    
    @property
    def models_dir(self) -> str:
        return self.get('paths.models_dir', './models')
    
    @property
    def moon_image(self) -> str:
        return self.get('paths.moon_image', 'moon.png')


# Global config instance
config = Config()