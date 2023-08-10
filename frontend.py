#/codespace/key-llme.py

import torch
from diffusers import DiffusionPipeline

def main(model_id=str, image_prompt=str):
    pipeline = DiffusionPipeline.from_pretrained(model_id).to("cuda")
    pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")
    rendered_image = pipeline(image_prompt).images[0]
    rendered_image.save("final_image.png")

if __name__ == "__main__":
    main("stabilityai/stable-diffusion-xl-base-1.0", "a small rock with sunglasses on")
