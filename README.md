# Joytag Caption - Batch
This tool utilizes the Joytag Caption tool (still in Pre-Alpha), to caption image files in a batch.

Place all images you wish to caption in the /input directory and run `py batch.py`.

Support for an `--img_dir` argument added by [CambridgeComputing](https://github.com/CambridgeComputing) It lets you specify a directory other than ./input. If no arguments are provided by the user, the script still defaults to the ./input subdirectory.

Now supports `LOW_VRAM_MODE=true`. This will use a [llama3-8b-bnb-4bit quantized version from unsloth](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit)

# Setup
1. Git clone this repository `git clone https://github.com/MNeMoNiCuZ/joy-caption-batch/`
2. (Optional) Create a virtual environment for your setup. Use python 3.9 to 3.11. Do not use 3.12. Feel free to use the `venv_create.bat` for a simple windows setup. Activate your venv.
3. Run `pip install -r requirements.txt` (this is done automatically with the `venv_create.bat`).
4. Install [PyTorch with CUDA support](https://pytorch.org/) matching your installed CUDA version. Run `nvcc --version` to find out which CUDA is your default.

You should now be set up and able to run the tool.

# Requirements
- Tested on Python 3.10 and 3.11.
- Tested on Pytorch w. CUDA 12.1.

> [!CAUTION]
> joy-caption requires a lot of VRAM. Make sure to turn on LOW_VRAM_MODE if you are below 24gb VRAM on your GPU.

# Example
![put_images_here](https://github.com/user-attachments/assets/a24251e5-6df6-44d4-a231-b74da9fcd8ca)
```
This image is a vibrant, detailed digital painting depicting a majestic golden dragon standing on a rocky outcrop in a lush, enchanted garden. The dragon, with its wings spread wide, has a regal, confident posture. Its scales are a shimmering gold, and its wings are a gradient of pink and purple hues, giving them a delicate, almost translucent appearance. The dragon's eyes are red and piercing, and its mouth is open, revealing sharp teeth.

In the background, a castle with tall, blue turrets emerges from the mist, adding an air of mystery and fantasy. The castle is partially obscured by trees and foliage, enhancing the sense of a magical, hidden realm. Surrounding the dragon are vibrant flowers in various stages of bloom, including pink and white lilies, blue hydrangeas, and yellow daisies, all adding to the enchanting atmosphere. The garden is lush and verdant, with green leaves and grasses, and the ground is a winding stone pathway that leads towards the castle.

The sky is a brilliant blue, dotted with fluffy white clouds, and the light is soft and diffused, creating a dreamlike quality to the entire scene. The overall style is highly detailed and realistic, with a touch of fantasy and whimsy.
```
![Man](https://github.com/user-attachments/assets/627fe206-fce3-43aa-8ede-b203dfa7f199)
![Woman](https://github.com/user-attachments/assets/6a22d1b5-8a28-42ea-a330-2886ab009aaf)


# Run the original online
Original app and source on huggingface: [https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha)


# Known issues
## Not using the right GPU?
You may need to set the CUDA device to GPU 0 by adding `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` near the start of the code the code.

## Too slow?
You may want to run the model in Low VRAM mode: Set `LOW_VRAM_MODE=true` in batch.py
