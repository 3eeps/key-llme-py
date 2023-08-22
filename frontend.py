#/codespace/key-llme.py
import torch
from datetime import datetime
from diffusers import DiffusionPipeline

def main():
    time_stamp = datetime.now()
    output_file_name = time_stamp.strftime("%d-%m-%Y-%H-%M")
    
    model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_path = "loras/Retro_rocket_sdxl.safetensors"
    
    pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipeline.enable_model_cpu_offload()
    pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict=lora_path, weight_name=lora_path)

    while True:
        user_prompt = input("key-llme-py>>> ")

        _image = pipeline(prompt=user_prompt).images[0]
        _image.save(f"{output_file_name}.png")

if __name__ == "__main__":
    main() 
