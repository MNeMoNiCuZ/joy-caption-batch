#!/usr/bin/env python3
"""
Use JoyCaption to caption images.
"""
import argparse
import dataclasses
import json
import logging
import os
import random
from pathlib import Path

import PIL.Image
import torch
import torch.amp
import torchvision.transforms.functional as TVF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlavaForConditionalGeneration,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def none_or_type(value, desired_type):
    if value == "None":
        return None
    return desired_type(value)

# Captioning Options and settings
# Prompt used for captioning
PROMPT = "Write a short description of the image. Include information about camera angle. Include information about whether there are JPEG artifacts or not."

"""
Example prompts

Short description
Write a short description of the image.

Detailed descriptive:
Please provide a detailed description of the image.

Descriptive:
Write a descriptive caption for this image in a formal tone.

Descriptive (Informal):
Write a descriptive caption for this image in a casual tone.

Training Prompt:
Write a stable diffusion prompt for this image.

MidJourney:
Write a MidJourney prompt for this image.

Booru tag list:
Write a list of Booru tags for this image.

Booru-like tag list:
Write a list of Booru-like tags for this image.

Art Critic:
Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.

Product Listing:
Write a caption for this image as though it were a product listing.

Social Media Post:
Write a caption for this image as if it were being used for a social media post.

Extra Options:
If there is a person/character in the image you must refer to them as {name}.
Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).
Include information about lighting.
Include information about camera angle.
Include information about whether there is a watermark or not.
Include information about whether there are JPEG artifacts or not.
If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.
Do NOT include anything sexual; keep it PG.
Do NOT mention the image's resolution.
You MUST include information about the subjective aesthetic quality of the image from low to very high.
Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.
Do NOT mention any text that is in the image.
Specify the depth of field and whether the background is in focus or blurred.
If applicable, mention the likely use of artificial or natural lighting sources.
Do NOT use any ambiguous language.
Include whether the image is sfw, suggestive, or nsfw.
ONLY describe the most important elements of the image.
"""


# Configuration options
PRINT_CAPTIONS = False  # Option to print captions to the console during inference
PRINT_CAPTIONING_STATUS = False  # Option to print captioning file status to the console
OVERWRITE = True  # Option to allow overwriting existing caption files
PREPEND_STRING = ""  # Prefix string to prepend to the generated caption
APPEND_STRING = ""  # Suffix string to append to the generated caption

# Specify input and output folder paths
SCRIPT_DIR = Path(__file__).parent
INPUT_FOLDER = SCRIPT_DIR / "input"
OUTPUT_FOLDER = INPUT_FOLDER

# LLM Settings
TEMPERATURE = 0.5  # Controls the randomness of predictions.
TOP_K = 10  # Limits the sampling pool to the top K most likely options at each step.
MAX_NEW_TOKENS = 300  # The maximum number of tokens to generate.

# Define supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif"}

# Define default prompt file
PROMPT_FILE = SCRIPT_DIR / "default_prompts.json"

parser = argparse.ArgumentParser(description="Batch caption images using JoyCaption.")

# Image source arguments (all optional)
parser.add_argument(
    "--glob",
    type=str,
    default=None,
    help="Glob pattern to find images (e.g., 'images/*.jpg').",
)
parser.add_argument(
    "--filelist",
    type=str,
    default=None,
    help="File containing list of image paths, one per line.",
)
parser.add_argument(
    "--input-folder",
    action="store_true",
    help=f"Process all images in the default input folder '{INPUT_FOLDER}'.",
)

# Prompt arguments (all optional)
parser.add_argument(
    "--prompt", type=str, default=PROMPT, help="Prompt to use for image captioning."
)
parser.add_argument(
    "--prompt-file",
    type=str,
    default=None,
    help=f"JSON file containing prompts to use. Default: '{PROMPT_FILE}'.",
)

# New optional argument for image_seq_length
parser.add_argument(
    "--image-seq-length",
    type=int,
    default=1,
    help="Number of image tokens to insert per image token in the prompt. Default is 1.",
)

# Other optional parameters with default values
parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
    help="Batch size for processing images.",
)
parser.add_argument(
    "--greedy", action="store_true", help="Use greedy decoding instead of sampling."
)
parser.add_argument(
    "--temperature",
    type=float,
    default=TEMPERATURE,
    help="Sampling temperature.",
)
parser.add_argument(
    "--top-p",
    type=lambda x: none_or_type(x, float),
    default=0.9,
    help="Top-p sampling.",
)
parser.add_argument(
    "--top-k",
    type=lambda x: none_or_type(x, int),
    default=TOP_K,
    help="Top-k sampling.",
)
parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=MAX_NEW_TOKENS,
    help="Maximum length of the generated caption (in tokens).",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=4,
    help="Number of workers loading images in parallel.",
)
parser.add_argument(
    "--model",
    type=str,
    default="fancyfeast/llama-joycaption-alpha-two-hf-llava",
    help="Model to use.",
)

# Additional optional parameters from configuration options
parser.add_argument(
    "--print-captions",
    action="store_true",
    default=PRINT_CAPTIONS,
    help="Print captions to the console during inference.",
)
parser.add_argument(
    "--print-captioning-status",
    action="store_true",
    default=PRINT_CAPTIONING_STATUS,
    help="Print captioning file status to the console.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    default=OVERWRITE,
    help="Allow overwriting existing caption files.",
)
parser.add_argument(
    "--prepend-string",
    type=str,
    default=PREPEND_STRING,
    help="Prefix string to prepend to the generated caption.",
)
parser.add_argument(
    "--append-string",
    type=str,
    default=APPEND_STRING,
    help="Suffix string to append to the generated caption.",
)

PIL.Image.MAX_IMAGE_PIXELS = 933120000  # Suppress Pillow warnings on large images


@dataclasses.dataclass
class Prompt:
    prompt: str
    weight: float


@torch.no_grad()
def main():
    # Logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    # Parse arguments
    args = parser.parse_args()
    logging.info(f"Arguments: {args}")

    # Determine if any image sources are provided
    image_sources_provided = any([args.glob, args.filelist, args.input_folder])

    # If no image sources are provided, default to using the input folder
    use_default_input = not image_sources_provided

    if use_default_input:
        logging.info("No image sources provided. Defaulting to the 'input' folder.")

    # Make sure we have a prompt or a prompt file
    prompts = parse_prompts(args.prompt, args.prompt_file)

    # Find the images
    image_paths = find_images(
        args.glob, args.filelist, args.input_folder, use_default_input
    )

    if len(image_paths) == 0:
        logging.warning(f"No images found in {image_paths}.")
        return
    logging.info(f"Total images found: {len(image_paths)}")

    # Handle overwrite option
    if not args.overwrite:
        # Ignore all images that already have captions
        image_paths = [
            path for path in image_paths if not Path(path).with_suffix(".txt").exists()
        ]
        logging.info(
            f"Images to caption after filtering existing captions: {len(image_paths)}"
        )
        if len(image_paths) == 0:
            logging.info("No new images to caption.")
            return
    else:
        logging.info("Overwrite option enabled; all images will be processed.")

    # Load JoyCaption
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(
        tokenizer, PreTrainedTokenizerFast
    ), f"Tokenizer is of type {type(tokenizer)}"
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="bfloat16", device_map="auto"
    )
    assert isinstance(llava_model, LlavaForConditionalGeneration)

    # Log image_seq_length for debugging
    logging.debug(f"Image sequence length: {args.image_seq_length}")

    dataset = ImageDataset(
        prompts,
        image_paths,
        tokenizer,
        llava_model.config.image_token_index,
        args.image_seq_length,  # Use the provided image_seq_length
    )
    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        batch_size=args.batch_size,
    )
    end_of_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    assert isinstance(end_of_header_id, int) and isinstance(end_of_turn_id, int)

    pbar = tqdm(
        total=len(image_paths), desc="Captioning images...", dynamic_ncols=True
    )
    for batch in dataloader:
        vision_dtype = (
            llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        )
        vision_device = (
            llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
        )
        language_device = (
            llava_model.language_model.get_input_embeddings().weight.device
        )

        # Move to GPU
        pixel_values = batch["pixel_values"].to(vision_device, non_blocking=True)
        input_ids = batch["input_ids"].to(language_device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(language_device, non_blocking=True)

        # Normalize the image
        pixel_values = pixel_values / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(vision_dtype)

        # Generate the captions
        try:
            generate_ids = llava_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy,
                suppress_tokens=None,
                use_cache=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            pbar.update(len(batch["paths"]))
            continue

        # Trim off the prompts
        assert isinstance(generate_ids, torch.Tensor)
        generate_ids = generate_ids.tolist()
        generate_ids = [
            trim_off_prompt(ids, end_of_header_id, end_of_turn_id)
            for ids in generate_ids
        ]

        # Decode the captions
        captions = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        captions = [c.strip() for c in captions]

        for path, caption in zip(batch["paths"], captions):
            write_caption(Path(path), caption, args)

        pbar.update(len(captions))
    pbar.close()
    logging.info("Captioning completed.")


def trim_off_prompt(input_ids: list[int], eoh_id: int, eot_id: int) -> list[int]:
    # Trim off the prompt
    while True:
        try:
            i = input_ids.index(eoh_id)
        except ValueError:
            break

        input_ids = input_ids[i + 1 :]

    # Trim off the end
    try:
        i = input_ids.index(eot_id)
    except ValueError:
        return input_ids

    return input_ids[:i]


def write_caption(image_path: Path, caption: str, args):
    caption_path = image_path.with_suffix(".txt")

    # Apply PREPEND_STRING and APPEND_STRING
    caption = f"{args.prepend_string}{caption}{args.append_string}"

    # If PRINT_CAPTIONS is True, print the caption to console
    if args.print_captions:
        print(f"Caption for '{image_path}': {caption}")

    # Handle OVERWRITE option
    if caption_path.exists():
        if args.overwrite:
            mode = "w"  # Overwrite existing file
        else:
            if args.print_captioning_status:
                print(
                    f"Caption file '{caption_path}' already exists and will not be overwritten."
                )
            return
    else:
        mode = "w"  # Create new file

    try:
        with open(caption_path, mode, encoding="utf-8") as f:
            f.write(caption)
        if args.print_captioning_status:
            print(f"Caption written to '{caption_path}'")
    except Exception as e:
        logging.error(f"Failed to write caption to '{caption_path}': {e}")


class ImageDataset(Dataset):
    def __init__(
        self,
        prompts: list[Prompt],
        paths: list[Path],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        image_token_id: int,
        image_seq_length: int,
    ):
        self.prompts = prompts
        self.paths = paths
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.image_seq_length = image_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]

        # Pick a prompt
        prompt_str = random.choices(
            self.prompts, weights=[p.weight for p in self.prompts]
        )[0].prompt

        # Preprocess image
        try:
            image = Image.open(path)
            if image.size != (384, 384):
                image = image.resize((384, 384), Image.LANCZOS)
            image = image.convert("RGB")
            pixel_values = TVF.pil_to_tensor(image)
        except Exception as e:
            logging.error(f"Failed to load image '{path}': {e}")
            pixel_values = None  # Will be filtered out later

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt_str,
            },
        ]

        # Format the conversation
        convo_string = self.tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(convo_string, str)

        # Tokenize the conversation
        convo_tokens = self.tokenizer.encode(
            convo_string, add_special_tokens=False, truncation=False
        )

        # Repeat the image tokens based on image_seq_length
        input_tokens = []
        for token in convo_tokens:
            if token == self.image_token_id:
                input_tokens.extend([self.image_token_id] * self.image_seq_length)
            else:
                input_tokens.append(token)

        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "path": path,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def collate_fn(self, batch: list[dict]) -> dict:
        # Filter out images that failed to load
        batch = [item for item in batch if item["pixel_values"] is not None]

        if not batch:
            return {
                "paths": [],
                "pixel_values": torch.empty(0),
                "input_ids": torch.empty(0),
                "attention_mask": torch.empty(0),
            }

        # Pad input_ids and attention_mask
        max_length = max(item["input_ids"].shape[0] for item in batch)
        n_pad = [max_length - item["input_ids"].shape[0] for item in batch]
        input_ids = torch.stack(
            [
                torch.nn.functional.pad(item["input_ids"], (n, 0), value=self.pad_token_id)
                for item, n in zip(batch, n_pad)
            ]
        )
        attention_mask = torch.stack(
            [
                torch.nn.functional.pad(item["attention_mask"], (n, 0), value=0)
                for item, n in zip(batch, n_pad)
            ]
        )

        # Stack pixel values
        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        # Paths
        paths = [item["path"] for item in batch]

        return {
            "paths": paths,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def parse_prompts(prompt_str: str | None, prompt_file: str | None) -> list[Prompt]:
    prompts = []
    if prompt_str:
        prompts.append(Prompt(prompt=prompt_str, weight=1.0))

    if prompt_file:
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            logging.error(f"Prompt file '{prompt_file}' does not exist.")
        else:
            try:
                data = json.loads(prompt_path.read_text())
                if not isinstance(data, list):
                    raise ValueError("Expected JSON file to contain a list of prompts.")

                for item in data:
                    if isinstance(item, str):
                        prompts.append(Prompt(prompt=item, weight=1.0))
                    elif (
                        isinstance(item, dict)
                        and "prompt" in item
                        and "weight" in item
                        and isinstance(item["prompt"], str)
                        and isinstance(item["weight"], (int, float))
                    ):
                        prompts.append(
                            Prompt(prompt=item["prompt"], weight=item["weight"])
                        )
                    else:
                        raise ValueError(
                            f"Invalid prompt in JSON file. Should be either a string or an object with 'prompt' and 'weight' fields: {item}"
                        )

            except json.JSONDecodeError as e:
                logging.error(
                    f"Failed to parse JSON from prompt file '{prompt_file}': {e}"
                )
            except ValueError as ve:
                logging.error(str(ve))

    if not prompts and PROMPT:
        logging.info(f"No prompts provided. Using default prompt: '{PROMPT}'")
        prompts.append(Prompt(prompt=PROMPT, weight=1.0))

    if not prompts:
        raise ValueError(
            "No prompts available. Please provide a prompt via --prompt or --prompt-file."
        )

    # Normalize weights
    total_weight = sum(p.weight for p in prompts)
    if total_weight <= 0.0:
        raise ValueError("Prompt weights must sum to a positive number.")

    return prompts


def find_images(
    glob: str | None,
    filelist: str | Path | None,
    input_folder_flag: bool,
    use_default_input: bool,
) -> list[Path]:
    paths = []

    if glob:
        glob_paths = list(Path(".").glob(glob))
        valid_glob_paths = [
            p for p in glob_paths if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        logging.info(f"Found {len(valid_glob_paths)} images using glob pattern '{glob}'.")
        paths.extend(valid_glob_paths)

    if filelist:
        filelist_path = Path(filelist)
        if not filelist_path.exists():
            logging.error(f"File list '{filelist}' does not exist.")
        else:
            filelist_paths = [
                Path(line.strip())
                for line in filelist_path.read_text().strip().splitlines()
                if line.strip() != ""
            ]
            valid_filelist_paths = [
                p for p in filelist_paths if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
            logging.info(
                f"Found {len(valid_filelist_paths)} images using file list '{filelist}'."
            )
            paths.extend(valid_filelist_paths)

    if input_folder_flag or use_default_input:
        if not INPUT_FOLDER.exists():
            logging.error(f"Default input folder '{INPUT_FOLDER}' does not exist.")
        elif not INPUT_FOLDER.is_dir():
            logging.error(f"Default input path '{INPUT_FOLDER}' is not a directory.")
        else:
            input_images = [
                p
                for p in INPUT_FOLDER.iterdir()
                if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
            if not input_images:
                logging.warning(f"No supported image files found in '{INPUT_FOLDER}'.")
            else:
                logging.info(
                    f"Adding {len(input_images)} images from the default input folder '{INPUT_FOLDER}'."
                )
                paths.extend(input_images)

    # Remove duplicates
    unique_paths = list(set(paths))

    # Log the final image paths for debugging
    logging.debug(f"Final image paths: {unique_paths}")

    return unique_paths


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
