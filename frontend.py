#/codespace/key-llme.py
import os
import torch
from datetime import datetime
from diffusers import DiffusionPipeline

def main():
    model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_path = "./loras"
    lora_list = []
    file_list = os.scandir(lora_path)
    
    print(model_path, "\n", lora_path)
    list_num = 0
    for obj in file_list:
        list_num = list_num + 1
        if obj.is_file():
            print(f"{list_num}.) {obj.name}")
            lora_list.append(obj.name)
    file_list.close()

    lora_num = int(input(">>> "))
    lora_to_load = lora_list[lora_num - 1]
    print(f"loading sdxl 1.0 + {lora_to_load}:")

    pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype=torch.float16)
    pipeline.enable_model_cpu_offload()
    pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict=lora_path, weight_name=lora_to_load)

    while True:
        user_prompt = input(">>> ") 
        _image = pipeline(prompt=user_prompt).images[0]
        output_file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")      
        _image.save(f"{output_file_name}.png")

if __name__ == "__main__":
    main() 
