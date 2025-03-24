import os
import torch
import numpy as np
from PIL import Image
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Model repository mapping
MODEL_REPOS = {
    "pyramid_flux": "rain1011/pyramid-flow-miniflux",
    "pyramid_sd3": "rain1011/pyramid-flow"
}

def download_model(model_name, local_dir):
    """
    Downloads the model from Hugging Face if not already present
    """
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        
    # Check if model files already exist
    if len(os.listdir(local_dir)) == 0:
        print(f"Downloading {model_name} model...")
        repo_id = MODEL_REPOS.get(model_name)
        if not repo_id:
            raise ValueError(f"Unknown model name: {model_name}")
            
        try:
            snapshot_download(
                repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                repo_type="model"
            )
            print(f"Successfully downloaded {model_name} to {local_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {str(e)}")
    else:
        print(f"Model files found in {local_dir}")

class PyramidDiTLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["pyramid_flux", "pyramid_sd3"], {"default": "pyramid_flux"}),
                "variant": (["diffusion_transformer_384p", "diffusion_transformer_768p"], {"default": "diffusion_transformer_384p"}),
                "model_path": ("STRING", {
                    "default": os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "models/pyramid-flow"),
                    "multiline": False
                }),
                "model_dtype": (["bf16"], {"default": "bf16"}),  # Currently only bf16 is supported
                "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False}),
                "enable_vae_tiling": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("PYRAMID_DIT_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"

    def load_model(self, model_name, variant, model_path, model_dtype, enable_sequential_cpu_offload, enable_vae_tiling):
        # Auto-download the model if needed
        try:
            download_model(model_name, model_path)
        except Exception as e:
            raise RuntimeError(f"Error downloading model: {str(e)}")

        # Create model instance
        try:
            model = PyramidDiTForVideoGeneration(
                model_path,
                model_name=model_name,
                model_dtype=model_dtype,
                model_variant=variant,
            )
        except Exception as e:
            raise RuntimeError(f"Error creating model: {str(e)}")

        # Handle device placement and optimizations
        try:
            if enable_sequential_cpu_offload:
                model.enable_sequential_cpu_offload()
            else:
                model.vae.to("cuda")
                model.dit.to("cuda")
                model.text_encoder.to("cuda")
                
            if enable_vae_tiling:
                model.vae.enable_tiling()
        except Exception as e:
            raise RuntimeError(f"Error configuring model: {str(e)}")
        
        return (model,)

class PyramidDiTText2Video:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("PYRAMID_DIT_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 640, "min": 384, "max": 2048}),
                "height": ("INT", {"default": 384, "min": 384, "max": 2048}),
                "num_frames": ("INT", {"default": 16, "min": 1, "max": 31}),  # temp=16 for 5s, temp=31 for 10s
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0}),  # 7-9 recommended
                "video_guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0}),
                "save_memory": ("BOOLEAN", {"default": True}),
                "cpu_offloading": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE_SEQUENCE",)
    FUNCTION = "generate"
    CATEGORY = "generators"

    def generate(self, model, prompt, width, height, num_frames, guidance_scale, video_guidance_scale, save_memory, cpu_offloading):
        try:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                frames = model.generate(
                    prompt=prompt,
                    num_inference_steps=[20, 20, 20],
                    video_num_inference_steps=[10, 10, 10],
                    height=height,
                    width=width,
                    temp=num_frames,
                    guidance_scale=guidance_scale,
                    video_guidance_scale=video_guidance_scale,
                    output_type="pil",
                    save_memory=save_memory,
                    cpu_offloading=cpu_offloading
                )
            
            # Convert PIL images to torch tensors
            frame_tensors = []
            for frame in frames:
                frame_tensor = torch.from_numpy(np.array(frame)).permute(2, 0, 1) / 255.0
                frame_tensors.append(frame_tensor)
            
            return (frame_tensors,)
        except Exception as e:
            raise RuntimeError(f"Error during video generation: {str(e)}")

class PyramidDiTImage2Video:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("PYRAMID_DIT_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "num_frames": ("INT", {"default": 16, "min": 1, "max": 31}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0}),
                "video_guidance_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0}),
                "save_memory": ("BOOLEAN", {"default": True}),
                "cpu_offloading": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE_SEQUENCE",)
    FUNCTION = "generate"
    CATEGORY = "generators"

    def generate(self, model, image, prompt, num_frames, guidance_scale, video_guidance_scale, save_memory, cpu_offloading):
        try:
            # Convert torch tensor to PIL Image
            image_pil = Image.fromarray((image.permute(1, 2, 0) * 255).numpy().astype(np.uint8))
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                frames = model.generate_i2v(
                    prompt=prompt,
                    input_image=image_pil,
                    num_inference_steps=[10, 10, 10],
                    temp=num_frames,
                    guidance_scale=guidance_scale,
                    video_guidance_scale=video_guidance_scale,
                    output_type="pil",
                    save_memory=save_memory,
                    cpu_offloading=cpu_offloading
                )
            
            # Convert PIL images to torch tensors
            frame_tensors = []
            for frame in frames:
                frame_tensor = torch.from_numpy(np.array(frame)).permute(2, 0, 1) / 255.0
                frame_tensors.append(frame_tensor)
            
            return (frame_tensors,)
        except Exception as e:
            raise RuntimeError(f"Error during video generation: {str(e)}")

class SaveVideoSequence:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE_SEQUENCE",),
                "filename": ("STRING", {"default": "output.mp4"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_video"
    CATEGORY = "output"
    OUTPUT_NODE = True

    def save_video(self, frames, filename, fps):
        try:
            # Convert torch tensors back to PIL images
            pil_frames = []
            for frame in frames:
                pil_frame = Image.fromarray((frame.permute(1, 2, 0) * 255).numpy().astype(np.uint8))
                pil_frames.append(pil_frame)
                
            export_to_video(pil_frames, filename, fps=fps)
            return ()
        except Exception as e:
            raise RuntimeError(f"Error saving video: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "PyramidDiTLoader": PyramidDiTLoader,
    "PyramidDiTText2Video": PyramidDiTText2Video,
    "PyramidDiTImage2Video": PyramidDiTImage2Video,
    "SaveVideoSequence": SaveVideoSequence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PyramidDiTLoader": "Load PyramidDiT Model",
    "PyramidDiTText2Video": "PyramidDiT Text to Video",
    "PyramidDiTImage2Video": "PyramidDiT Image to Video",
    "SaveVideoSequence": "Save Video Sequence",
}
