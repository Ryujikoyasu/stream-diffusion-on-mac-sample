from typing import Literal

import torch

from .utils import StreamDiffusionWrapper


class StreamDiffusion:
    def __init__(
        self,
        model_id_or_path: str = "stabilityai/sd-turbo",
        prompt: str = "moon,impressionist painting style",
        negative_prompt: str = """
            human, person, people, face, portrait, girl, boy, man, woman, child, 
            body, hands, fingers, nsfw, figure, silhouette, shadow person, 
            character, anime character, manga, anime style, 
            humanoid, anthropomorphic, human-like, human shape, 
            facial features, eyes, mouth, nose, hair, 
            clothing, dress, outfit, costume,
            standing figure, walking figure, sitting figure,
            1girl, 2girls, multiple girls, multiple people,
            human elements, human presence, human form
        """.replace('\n', '').replace(' ', ''),
        acceleration: Literal["none", "xformers", "tensorrt"] = "none",
        use_denoising_batch: bool = True,
        use_tiny_vae: bool = True,
        guidance_scale: float = 1.5,
        delta: float = 5.0,
    ) -> None:
        self.prompt = prompt
        
        # MPSデバイスの最適化設定
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            # メモリ使用量の最適化
            torch.mps.set_per_process_memory_fraction(0.8)
            torch.mps.empty_cache()
            print("Using MPS device with optimized settings")
        else:
            self.device = torch.device("cpu")
            print("MPS not available, using CPU")
        
        try:
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=model_id_or_path,
                t_index_list=[8],
                warmup=1,
                device=self.device,
                acceleration=acceleration,
                mode="img2img",
                use_denoising_batch=use_denoising_batch,
                use_tiny_vae=use_tiny_vae,
                cfg_type="none",
            )
            
            # half精度の設定（必要な場合）
            if hasattr(self.stream, 'unet'):
                self.stream.unet = self.stream.unet.half()
            if hasattr(self.stream, 'vae'):
                self.stream.vae = self.stream.vae.half()
            
            self.stream.prepare(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=16,
                guidance_scale=guidance_scale,
                delta=delta,
            )
            
        except Exception as e:
            print(f"モデル初期化エラー: {e}")
            raise

    def _initialize_stream(self):
        """ストリームの再初期化（必要な場合）"""
        if self.stream is not None:
            torch.mps.empty_cache()
            # 個別のコンポーネントに対してhalf精度を適用
            if hasattr(self.stream, 'unet'):
                self.stream.unet = self.stream.unet.half()
            if hasattr(self.stream, 'vae'):
                self.stream.vae = self.stream.vae.half()
        return self.stream
