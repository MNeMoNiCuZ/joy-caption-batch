import spaces
import gradio as gr
from huggingface_hub import InferenceClient
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM, logging as transformers_logging
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import requests

# Configuration options
LOW_VRAM_MODE = True  # Option to switch to a model that uses less VRAM
PRINT_CAPTIONS = False  # Option to print captions to the console during inference
PRINT_CAPTIONING_STATUS = False  # Option to print captioning file status to the console
OVERWRITE = True  # Option to allow overwriting existing caption files
PREPEND_STRING = ""  # Prefix string to prepend to the generated caption
APPEND_STRING = ""  # Suffix string to append to the generated caption
BATCH_PROCESSING_COUNT = 4  # Number of images to process in a batch
#On a 3090 (24gb vram), you can use a batch count of 6.
#On a 3090 (24gb vram) with low_vram enabled, you can use a batch count of 16.

# Specify input and output folder paths
INPUT_FOLDER = Path(__file__).parent / "input"
OUTPUT_FOLDER = INPUT_FOLDER

# LLM Settings
VLM_PROMPT = "A descriptive caption for this image:\n"  # Changing this won't change the outputs. It's only here as a template.
TEMPERATURE = 0.5  # Controls the randomness of predictions. Lower values make the output more focused and deterministic, while higher values increase randomness.
TOP_K = 10  # Limits the sampling pool to the top K most likely options at each step. A lower value makes the output more deterministic, while a higher value allows more diversity.
MAX_NEW_TOKENS = 300  # The maximum number of tokens to generate. This limits the length of the generated text.

# Clip path
CLIP_PATH = "google/siglip-so400m-patch14-384"
CHECKPOINT_PATH = Path("wpkklhc6")
TITLE = "<h1><center>JoyCaption Pre-Alpha (2024-07-30a)</center></h1>"

# Model paths based on VRAM usage
if LOW_VRAM_MODE:
    MODEL_PATH = "unsloth/llama-3-8b-bnb-4bit"
else:
    MODEL_PATH = "unsloth/Meta-Llama-3.1-8B"

# Suppress warnings if PRINT_CAPTIONING_STATUS is False
if not PRINT_CAPTIONING_STATUS:
    transformers_logging.set_verbosity_error()

print("Captioning Batch Images Initializing...")

HF_TOKEN = os.environ.get("HF_TOKEN", None)

# URL for downloading the image adapter
IMAGE_ADAPTER_URL = "https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/blob/main/wpkklhc6/image_adapter.pt"

# Function to download the image adapter from a Hugging Face Space
def download_image_adapter(force_download=False):
    file_path = CHECKPOINT_PATH / "image_adapter.pt"
    if force_download or not file_path.exists():
        print(f"Downloading {file_path.name} from Hugging Face Space...")
        url = "https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/resolve/main/wpkklhc6/image_adapter.pt"
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {file_path.name} successfully.")
        else:
            print(f"Failed to download {file_path.name}. Status code: {response.status_code}")
            exit(1)  # Exit if download fails
    else:
        print(f"{file_path.name} already exists.")

# Download the image adapter before proceeding
download_image_adapter()


# Class definition for ImageAdapter
class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
    
    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x

# Process all images in the input folder recursively
print("Captioning Initializing")
image_files = list(INPUT_FOLDER.rglob('*'))

# Filter the list based on the Overwrite flag
if not OVERWRITE:
    image_files = [
        image_path for image_path in image_files
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'] and not (image_path.parent / (image_path.stem + ".txt")).exists()
    ]
else:
    image_files = [
        image_path for image_path in image_files
        if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    ]

total_images = len(image_files)
print(f"Found {total_images} files to process.")

if not image_files:
    print("No images to process. Exiting...")
    exit(0)  # Exit the script if there are no images to process

# Load CLIP, model, and other resources only if there are images to process
print("Loading CLIP")
clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
clip_model = AutoModel.from_pretrained(CLIP_PATH)
clip_model = clip_model.vision_model
clip_model.eval()
clip_model.requires_grad_(False)
clip_model.to("cuda")

# Tokenizer
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

# LLM
print("Loading LLM")
text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
text_model.eval()

# Image Adapter
print("Loading image adapter")
try:
    image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size)
    image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu"))
    image_adapter.eval()
    image_adapter.to("cuda")
except (torch.nn.modules.module.ModuleAttributeError, _pickle.UnpicklingError):
    print("The image adapter file is corrupted. Re-downloading...")
    # Force re-download
    download_image_adapter(force_download=True)
    # Try loading again
    image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu"))
    image_adapter.eval()
    image_adapter.to("cuda")

@spaces.GPU()
@torch.no_grad()
def process_images_in_batch(batch_paths):
    torch.cuda.empty_cache()

    # Preprocess images in batch
    images = []
    for path in batch_paths:
        input_image = Image.open(path).convert("RGB")
        images.append(clip_processor(images=input_image, return_tensors='pt').pixel_values)
    images = torch.cat(images, dim=0).to('cuda')

    # Tokenize the prompt
    prompt = tokenizer.encode(VLM_PROMPT, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)

    # Embed images
    with torch.amp.autocast_mode.autocast('cuda', enabled=True):
        vision_outputs = clip_model(pixel_values=images, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[-2]
        embedded_images = image_adapter(image_features)

    # Embed prompt
    prompt_embeds = text_model.model.embed_tokens(prompt.to('cuda'))
    embedded_bos = text_model.model.embed_tokens(
        torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64)
    )

    # Construct prompts
    inputs_embeds = torch.cat([
        embedded_bos.expand(embedded_images.shape[0], -1, -1),
        embedded_images.to(dtype=embedded_bos.dtype),
        prompt_embeds.expand(embedded_images.shape[0], -1, -1),
    ], dim=1)

    input_ids = torch.cat([
        torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).expand(len(batch_paths), -1),
        torch.zeros((len(batch_paths), embedded_images.shape[1]), dtype=torch.long),
        prompt.expand(len(batch_paths), -1),
    ], dim=1).to('cuda')
    attention_mask = torch.ones_like(input_ids)

    # Generate captions for the batch
    generate_ids = text_model.generate(
        input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_k=TOP_K,
        temperature=TEMPERATURE,
        suppress_tokens=None
    )

    # Process captions
    captions = []
    for idx, image_path in enumerate(batch_paths):
        generated_ids = generate_ids[idx, input_ids.shape[1]:]
        if generated_ids[-1] == tokenizer.eos_token_id:
            generated_ids = generated_ids[:-1]

        caption = f"{PREPEND_STRING}{tokenizer.decode(generated_ids, skip_special_tokens=True)}{APPEND_STRING}"
        captions.append(caption)

        # Save caption to text file
        output_file_path = image_path.parent / (image_path.stem + ".txt")
        with open(output_file_path, 'w') as caption_file:
            caption_file.write(caption)
        if PRINT_CAPTIONS:
            print(f"Caption for {image_path}: {caption}")

    return captions

# Process images in batches
batch_start = 0
captions_all = []
with tqdm(total=total_images, unit='image') as pbar:
    while batch_start < len(image_files):
        batch_end = min(batch_start + BATCH_PROCESSING_COUNT, len(image_files))
        batch_paths = image_files[batch_start:batch_end]
        
        captions = process_images_in_batch(batch_paths)
        captions_all.extend(captions)
        
        batch_start = batch_end
        pbar.update(len(batch_paths))

print(f"Finished processing {len(captions_all)} captions.")
