#/codespace/key-llme.py
import torch
from datetime import datetime
from diffusers import DiffusionPipeline

def main():
    time_stamp = datetime.now()
    output_file_name = time_stamp.strftime("%d-%m-%Y-%H-%M")
    
    model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_path = "fr4z3tt4.safetensors"

    pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype=torch.float16).to("cuda")
    pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict=lora_path, weight_name=lora_path)

    while True:
        prompt = input("key-llme-py>>> ")

        _image = pipeline(prompt).images[0]
        _image.save(f"{output_file_name}.png")

if __name__ == "__main__":
    main()
