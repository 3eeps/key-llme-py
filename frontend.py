#/codespace/key-llme.py
import torch
from datetime import datetime
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

def main():
    timestamp = datetime.now()
    file_name = timestamp.strftime("%d-%m-%Y-%H-%M")

    model_base = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_home = "nerijs/pixel-art-xl"
    sf_weights = "pixel-art-xl.safetensors"

    pipeline = DiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16, use_safetensors=True, local_files_only=True, safety_check=None).to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights(lora_home, weight_name=sf_weights)

    user_prompt = input("key-llme-py>>> ")
    with torch.no_grad():
        _image_seed = torch.Generator("cuda").manual_seed(0)
        _image = pipeline(user_prompt, generator=_image_seed, num_inference_steps=50).images[0]

    _image.save(f"{file_name}.png")

if __name__ == "__main__":
    main()
