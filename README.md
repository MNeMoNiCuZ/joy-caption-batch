# Joytag Caption - Batch
This tool utilizes the Joytag Caption tool (still in Pre-Alpha), to caption image files in a batch.

Place all images you wish to caption in the /input directory and run `py batch.py`.

# Setup
1. Git clone this repository `git clone https://github.com/MNeMoNiCuZ/joy-caption-batch/`
2. Download the [image_adapter.pt](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/tree/main/wpkklhc6) and place it in the `wpkklhc6` directory.
3. (Optional) Create a virtual environment for your setup. Feel free to use the `venv_create.bat` for a simple windows setup. Activate your venv.
5. Run `pip install -r requirements.txt`
6. Install [PyTorch with CUDA support](https://pytorch.org/)

# Requirements

- Tested on Python 3.10 and 3.12.
- Tested on Pytorch w. CUDA 12.1.
- Should also work on other versions.

# Run the original online
Original app and source on huggingface: [https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha)
