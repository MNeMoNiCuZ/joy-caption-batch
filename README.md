# Joytag Caption - Batch
This tool utilizes the Joytag Caption tool (still in Pre-Alpha), to caption image files in a batch.

Simply place all images you wish to caption in the /input directory and run `py batch.py`.


# Setup
1. Download the [image_adapter.pt](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/tree/main/wpkklhc6) and place it in the `wpkklhc6` directory.
2. (Optional) Create a virtual environment for your setup. Feel free to use the `venv_create.bat` for a simple windows setup.
3. (Optional) Activate your venv and run `pip install -r requirements.txt`


Original Readme
---
title: Joy Caption Pre Alpha
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).
