import torch
import os
import numpy as np
from PIL import Image
import folder_paths
from transformers import AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

class OvisLoader:
    def __init__(self):
        self.loaded_model = None
        self.model_name = 'AIDC-AI/Ovis1.6-Llama3.2-3B'

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["cuda", "cpu"],),
            }
        }

    RETURN_TYPES = ("OVIS_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Ovis"

    def load_model(self, device):
        if self.loaded_model is None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                multimodal_max_length=8192,
                trust_remote_code=True
            ).to(device=device)

            self.loaded_model = {
                "model": model,
                "text_tokenizer": model.get_text_tokenizer(),
                "visual_tokenizer": model.get_visual_tokenizer()
            }

        return (self.loaded_model,)

class OvisInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ovis_model": ("OVIS_MODEL",),
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 2048}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "Ovis"

    def generate(self, ovis_model, image, text, max_new_tokens):
        model = ovis_model["model"]
        text_tokenizer = ovis_model["text_tokenizer"]
        visual_tokenizer = ovis_model["visual_tokenizer"]

        # Convert ComfyUI tensor to PIL Image
        if image is not None:
            # Remove batch dimension if present and ensure we're working with the first image
            if len(image.shape) == 4:
                image = image[0]

            # Convert from HWC to CHW format
            image = image.permute(2, 0, 1)

            # Scale to 0-255 range and convert to uint8
            image = (image * 255).clamp(0, 255).byte()

            # Convert to numpy array in HWC format
            image = image.permute(1, 2, 0).cpu().numpy()

            # Create PIL Image
            image = Image.fromarray(image, mode='RGB')

        # Prepare conversation format
        conversations = [
            {"from": "human", "value": "<image>\n" + text if image is not None else text}
        ]

        # Prepare inputs
        prompt, input_ids, pixel_values = model.preprocess_inputs(conversations, [image])
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)

        if image is None:
            pixel_values = [None]
        else:
            pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

        # Setup generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "top_p": None,
            "top_k": None,
            "temperature": None,
            "repetition_penalty": None,
            "eos_token_id": model.generation_config.eos_token_id,
            "pad_token_id": text_tokenizer.pad_token_id,
            "use_cache": True
        }

        # Generate response
        streamer = TextIteratorStreamer(text_tokenizer, skip_prompt=True, skip_special_tokens=True)
        response = ""

        thread = Thread(
            target=model.generate,
            kwargs={
                "inputs": input_ids,
                "pixel_values": pixel_values,
                "attention_mask": attention_mask,
                "streamer": streamer,
                **gen_kwargs
            }
        )
        thread.start()

        for new_text in streamer:
            response += new_text
        thread.join()

        return (response,)

NODE_CLASS_MAPPINGS = {
    "OvisLoader": OvisLoader,
    "OvisInference": OvisInference
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OvisLoader": "Load Ovis Model",
    "OvisInference": "Ovis Generate"
}
