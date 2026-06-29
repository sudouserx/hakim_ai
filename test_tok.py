from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("wnkh/llava-med-v1.5-mistral-7b-hf")
text = "<image>" * 576
tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
print("Number of <image> tokens:", tokens.count(32000))
print("Total tokens:", len(tokens))
print("First 10 tokens:", tokens[:10])
