#/codespace/key-llme.py
import torch
from datetime import datetime
from diffusers import DiffusionPipeline

def main(model_base=str):
    timestamp = datetime.now()
    file_name = timestamp.strftime("%d-%m-%Y-%H-%M-%S")

    image_prompt = input(">>> ")

    model_base = "stabilityai/stable-diffusion-xl-base-1.0"
    pipeline = DiffusionPipeline.from_pretrained(model_base).to("cuda")
    pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")

    rendered_image = pipeline(image_prompt).images[0]
    rendered_image.save(f"{file_name}.png")

if __name__ == "__main__":
    main()
