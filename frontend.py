#/codespace/key-llme.py
import os
import torch
from datetime import datetime
from diffusers import DiffusionPipeline

def main():
    time_stamp = datetime.now()
    output_file_name = time_stamp.strftime("%d-%m-%Y-%H-%M")
    
    model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    lora_path = "/codespace/loras"
    file_list = os.scandir(lora_path)
    
    print("# load lora")
    iter = 0
    lora_list = []
    for obj in file_list:
        iter = iter + 1
        if obj.is_file():
            print(f"{iter}.) {obj.name}")
            lora_list.append(obj.name)
    file_list.close()
    lora_num = int(input("#>>> "))
    lora_to_load = lora_list[lora_num - 1]
    print(f"sdxl 1.0 + {lora_to_load}")

    pipeline = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path=model_path, torch_dtype=torch.float16)
    pipeline.enable_model_cpu_offload()
    pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict=lora_path, weight_name=lora_to_load)

    while True:
        user_prompt = input("%>>> ")
        
        _image = pipeline(prompt=user_prompt).images[0]
        _image.save(f"{output_file_name}.png")

if __name__ == "__main__":
    main() 
