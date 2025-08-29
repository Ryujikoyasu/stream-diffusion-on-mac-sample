from typing import Literal

import torch

from .utils import StreamDiffusionWrapper


class StreamDiffusion:
    def __init__(
        self,
        model_id_or_path: str = "stabilityai/sd-turbo",
        prompt: str = "moon",
        negative_prompt: str = "human, person, people, face, portrait, girl, boy, man, woman, child, body, hands, fingers, nsfw, figure, silhouette, shadow person, character, anime character, manga, anime style, humanoid, anthropomorphic, human-like, human shape, facial features, eyes, mouth, nose, hair, clothing, dress, outfit, costume,standing figure, walking figure, sitting figure,1girl, 2girls, multiple girls, multiple people,human elements, human presence, human form".replace('\n', '').replace(' ', ''),
        acceleration: Literal["none", "xformers", "tensorrt"] = "none",
        use_denoising_batch: bool = True,
        use_tiny_vae: bool = True,
        guidance_scale: float = 0.6,
        delta: float = 1.5,
        local_cache_dir: str = "./models",
        # ref版の設定を追加（互換性オプション）
        optimize_for_speed: bool = True,
        use_kohaku_model: bool = False,
    ) -> None:
        self.prompt = prompt
        
        # デバイス設定（StreamDiffusionのCUDAイベント問題を回避するためCPUを使用）
        # TODO: MPSサポートのためにStreamDiffusionライブラリの修正が必要
        self.device = torch.device("mps")
        print("🖥️ Using CPU (MPS has compatibility issues with StreamDiffusion)")
        
        # モデル選択（ref版の芸術特化モデルをオプションで）
        if use_kohaku_model:
            model_id_or_path = "KBlueLeaf/kohaku-v2.1"
            print("🎨 Using Kohaku artistic model")
        
        # 速度最適化設定
        if optimize_for_speed:
            t_index_list = [8]
            warmup = 1
            num_inference_steps = 16
            print("⚡ Speed optimization enabled")
        else:
            t_index_list = [8]
            warmup = 10
            num_inference_steps = 16
        
        try:
            # StreamDiffusionWrapper expects device string, convert from device object
            if self.device.type == "cuda":
                device_str = "cuda"
            elif self.device.type == "mps":
                device_str = "mps"
            else:
                device_str = "cpu"
            
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=model_id_or_path,
                t_index_list=t_index_list,
                warmup=warmup,
                device=device_str,
                acceleration=acceleration,
                mode="img2img",
                use_denoising_batch=use_denoising_batch,
                use_tiny_vae=use_tiny_vae,
                cfg_type="none" if optimize_for_speed else ("self" if guidance_scale > 1.0 else "none"),
                seed=-1,  # ランダムシード有効化
                local_cache_dir=local_cache_dir,
            )
            
            # CPU使用時はfloat32のまま（half precisionはCPUで未サポート）
            print("🔧 Using float32 precision for CPU compatibility")
            
            self.stream.prepare(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                delta=delta,
            )
            
        except Exception as e:
            print(f"❌ モデル初期化エラー: {e}")
            raise

    def _initialize_stream(self):
        """ストリームの再初期化（ref版からの追加機能）"""
        if self.stream is not None and self.device.type == "mps":
            torch.mps.empty_cache()
            # 個別のコンポーネントに対してhalf精度を適用
            if hasattr(self.stream, 'unet'):
                self.stream.unet = self.stream.unet.half()
            if hasattr(self.stream, 'vae'):
                self.stream.vae = self.stream.vae.half()
            print("🔄 Stream reinitialized with optimizations")
        return self.stream
