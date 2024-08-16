import spaces
import gradio as gr
from huggingface_hub import InferenceClient
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os


CLIP_PATH = "google/siglip-so400m-patch14-384"
VLM_PROMPT = "A descriptive caption for this image:\n"
MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B"
CHECKPOINT_PATH = Path("wpkklhc6")
TITLE = "<h1><center>JoyCaption Pre-Alpha (2024-07-30a)</center></h1>"

HF_TOKEN = os.environ.get("HF_TOKEN", None)


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


# Load CLIP
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
image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size)
image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu"))
image_adapter.eval()
image_adapter.to("cuda")


@spaces.GPU()
@torch.no_grad()
def stream_chat(input_image: Image.Image):
	torch.cuda.empty_cache()

	# Preprocess image
	image = clip_processor(images=input_image, return_tensors='pt').pixel_values
	image = image.to('cuda')

	# Tokenize the prompt
	prompt = tokenizer.encode(VLM_PROMPT, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)

	# Embed image
	with torch.amp.autocast_mode.autocast('cuda', enabled=True):
		vision_outputs = clip_model(pixel_values=image, output_hidden_states=True)
		image_features = vision_outputs.hidden_states[-2]
		embedded_images = image_adapter(image_features)
		embedded_images = embedded_images.to('cuda')
	
	# Embed prompt
	prompt_embeds = text_model.model.embed_tokens(prompt.to('cuda'))
	assert prompt_embeds.shape == (1, prompt.shape[1], text_model.config.hidden_size), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], text_model.config.hidden_size)}"
	embedded_bos = text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64))

	# Construct prompts
	inputs_embeds = torch.cat([
		embedded_bos.expand(embedded_images.shape[0], -1, -1),
		embedded_images.to(dtype=embedded_bos.dtype),
		prompt_embeds.expand(embedded_images.shape[0], -1, -1),
	], dim=1)

	input_ids = torch.cat([
		torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
		torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
		prompt,
	], dim=1).to('cuda')
	attention_mask = torch.ones_like(input_ids)

	#generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=False, suppress_tokens=None)
	generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=True, top_k=10, temperature=0.5, suppress_tokens=None)

	# Trim off the prompt
	generate_ids = generate_ids[:, input_ids.shape[1]:]
	if generate_ids[0][-1] == tokenizer.eos_token_id:
		generate_ids = generate_ids[:, :-1]

	caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

	return caption.strip()


with gr.Blocks() as demo:
	gr.HTML(TITLE)
	with gr.Row():
		with gr.Column():
			input_image = gr.Image(type="pil", label="Input Image")
			run_button = gr.Button("Caption")
		
		with gr.Column():
			output_caption = gr.Textbox(label="Caption")
	
	run_button.click(fn=stream_chat, inputs=[input_image], outputs=[output_caption])


if __name__ == "__main__":
    demo.launch()