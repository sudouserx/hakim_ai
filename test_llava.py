import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image

model_name = "wnkh/llava-med-v1.5-mistral-7b-hf"
processor = LlavaProcessor.from_pretrained(model_name)
processor.patch_size = 14
processor.vision_feature_select_strategy = "default"

image = Image.new("RGB", (336, 336), color="red")
prompt = "USER: <image>\nDescribe this.\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")
print("Input IDs shape:", inputs["input_ids"].shape)
image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
print("Number of <image> tokens in input_ids:", (inputs["input_ids"] == image_token_id).sum().item())
