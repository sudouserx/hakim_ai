import urllib.request
url = "https://raw.githubusercontent.com/huggingface/transformers/v4.40.0/src/transformers/models/llava/processing_llava.py"
urllib.request.urlretrieve(url, "processing_llava.py")
