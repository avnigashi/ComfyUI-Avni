import os
import torch
import numpy as np
import folder_paths
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from torch.hub import download_url_to_file
from enum import Enum
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from transformers import T5EncoderModel, T5Tokenizer
import torch.nn.functional as F
from diffusers import DDIMScheduler

class CogVideoXModel(Enum):
    COGVIDEOX_2B = "THUDM/CogVideoX-2b"
    COGVIDEOX_5B = "THUDM/CogVideoX-5b"
    COGVIDEOX_5B_I2V = "THUDM/CogVideoX-5b-I2V"

@dataclass
class ModelSpecs:
    name: CogVideoXModel
    default_dtype: torch.dtype
    min_memory: int
    recommended_precision: str
    max_resolution: Tuple[int, int]
    fps: int
    video_length: int

class ModelRegistry:
    SPECS = {
        CogVideoXModel.COGVIDEOX_2B: ModelSpecs(
            name=CogVideoXModel.COGVIDEOX_2B,
            default_dtype=torch.float16,
            min_memory=4,
            recommended_precision="FP16",
            max_resolution=(720, 480),
            fps=8,
            video_length=6
        ),
        CogVideoXModel.COGVIDEOX_5B: ModelSpecs(
            name=CogVideoXModel.COGVIDEOX_5B,
            default_dtype=torch.bfloat16,
            min_memory=5,
            recommended_precision="BF16",
            max_resolution=(720, 480),
            fps=8,
            video_length=6
        ),
        CogVideoXModel.COGVIDEOX_5B_I2V: ModelSpecs(
            name=CogVideoXModel.COGVIDEOX_5B_I2V,
            default_dtype=torch.bfloat16,
            min_memory=5,
            recommended_precision="BF16",
            max_resolution=(720, 480),
            fps=8,
            video_length=6
        )
    }

    @classmethod
    def get_specs(cls, model: CogVideoXModel) -> ModelSpecs:
        return cls.SPECS[model]


class VideoProcessor:
    @staticmethod
    def prepare_image(image: torch.Tensor, target_size: Tuple[int, int]) -> Image.Image:
        # Convert ComfyUI tensor format
        image = torch.clamp((image[0].permute(1, 2, 0) + 1.0) / 2.0, min=0.0, max=1.0)
        image = (image * 255).cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(image)

        # Resize while maintaining aspect ratio
        aspect_ratio = min(target_size[0] / pil_image.size[0], target_size[1] / pil_image.size[1])
        new_size = tuple(int(dim * aspect_ratio) for dim in pil_image.size)
        new_size = (new_size[0] - (new_size[0] % 8), new_size[1] - (new_size[1] % 8))
        return pil_image.resize(new_size, Image.LANCZOS)


    @staticmethod
    def process_frames(frames: List[Image.Image]) -> torch.Tensor:
        processed_frames = []
        for frame in frames:
            frame_tensor = torch.from_numpy(np.array(frame)).float() / 127.5 - 1
            frame_tensor = frame_tensor.permute(2, 0, 1)
            processed_frames.append(frame_tensor)
        return torch.stack(processed_frames)

class CacheManager:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.lora_path = os.path.join(base_path, "lora")
        os.makedirs(self.lora_path, exist_ok=True)

    def get_lora_weights(self, orbit_type: str) -> str:
        filename = f"orbit_{orbit_type.lower()}_lora_weights.safetensors"
        local_path = os.path.join(self.lora_path, filename)

        if not os.path.exists(local_path):
            url = f"https://huggingface.co/wenqsun/DimensionX/resolve/main/{filename}"
            download_url_to_file(url, local_path)

        return local_path

class PromptManager:
    ENHANCEMENTS = [
        "high quality",
        "ultrarealistic detail",
        "cinematic lighting",
        "breath-taking movie-like camera shot"
    ]

    @classmethod
    def enhance_prompt(cls, prompt: str) -> str:
        return f"{prompt}. {', '.join(cls.ENHANCEMENTS)}."

class AdvancedCogVideoXNode:
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.cache_manager = CacheManager(os.path.join(folder_paths.models_dir, "cogvideox"))
        self.video_processor = VideoProcessor()
        self.prompt_manager = PromptManager()
        self.current_model = None
        self.pipe = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (
                    [model.value for model in CogVideoXModel],
                    {"default": CogVideoXModel.COGVIDEOX_5B_I2V.value}
                ),
                "precision": (
                    ["FP16", "BF16", "FP32", "INT8"],
                    {"default": "BF16"}
                ),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "orbit_type": (["Left", "Up"],),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0}),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True}),
                "enable_optimization": ("BOOLEAN", {"default": True}),
                "enable_vae_slicing": ("BOOLEAN", {"default": True}),
                "enable_vae_tiling": ("BOOLEAN", {"default": True}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "generate"
    CATEGORY = "video"

    def initialize_model(self, model_version: str, precision: str, enable_optimization: bool):
        try:
            model = CogVideoXModel(model_version)
            specs = self.model_registry.get_specs(model)

            dtype = {
                "FP16": torch.float16,
                "BF16": torch.bfloat16,
                "FP32": torch.float32,
                "INT8": torch.int8
            }[precision]

            if precision == "INT8":
                pipe = self._initialize_quantized_pipeline(model.value, dtype)
            else:
                pipe = self._initialize_standard_pipeline(model.value, dtype)

            return pipe, specs

        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise


    def _initialize_quantized_pipeline(self, model_id: str, dtype: torch.dtype):
        from torchao.quantization import quantize_, int8_weight_only

        text_encoder = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype)
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=dtype)
        vae = AutoencoderKLCogVideoX.from_pretrained(
            model_id, subfolder="vae", torch_dtype=dtype)

        for model in [text_encoder, transformer, vae]:
            quantize_(model, int8_weight_only())

        return CogVideoXImageToVideoPipeline.from_pretrained(
            model_id,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            torch_dtype=dtype
        )



    def _handle_lora_weights(self, pipe: CogVideoXImageToVideoPipeline, orbit_type: str, lora_scale: float = 1.0):
        lora_path = self.cache_manager.get_lora_weights(orbit_type)
        pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=os.path.basename(lora_path))
        pipe.fuse_lora(lora_scale=lora_scale / 256)  # Using default lora_rank of 256

    def _initialize_standard_pipeline(self, model_id: str, dtype: torch.dtype):
        text_encoder = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder",
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to("cuda")

        transformer = CogVideoXTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer",
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to("cuda")

        vae = AutoencoderKLCogVideoX.from_pretrained(
            model_id, subfolder="vae",
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to("cuda")

        tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

        try:
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                model_id,
                text_encoder=text_encoder,
                transformer=transformer,
                vae=vae,
                tokenizer=tokenizer,
                scheduler=scheduler,
                torch_dtype=dtype
            )
        except Exception as e:
            print(f"Error creating pipeline: {str(e)}")
            raise

        return pipe

    def generate(self, model_version: str, precision: str, image: torch.Tensor, prompt: str,
                 orbit_type: str, steps: int, cfg: float, seed: int, **kwargs):
        pipe = None
        try:
            pipe = self.initialize_model(model_version, precision, enable_optimization=True)

            # Handle optimizations
            if kwargs.get('enable_vae_slicing', True):
                pipe.vae.enable_slicing()
            if kwargs.get('enable_vae_tiling', True):
                pipe.vae.enable_tiling()

            # Prepare image
            pil_image = self.video_processor.prepare_image(image, (720, 480))

            # Generate
            generator = torch.Generator("cuda").manual_seed(seed if seed != -1 else random.randint(0, 2**32-1))

            with torch.inference_mode():
                result = pipe(
                    pil_image,
                    prompt=self.prompt_manager.enhance_prompt(prompt),
                    negative_prompt=kwargs.get('negative_prompt', ''),
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator
                )

            video_tensor = self.video_processor.process_frames(result.frames[0])
            motion_mask = self._generate_motion_mask(video_tensor)

            return (video_tensor, motion_mask)

        except Exception as e:
            print(f"Generation error: {str(e)}")
            raise
        finally:
            if pipe:
                pipe.to("cpu")
                torch.cuda.empty_cache()
    

NODE_CLASS_MAPPINGS = {
    "AdvancedCogVideoX": AdvancedCogVideoXNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedCogVideoX": "Advanced CogVideoX (Multi-Model)"
}