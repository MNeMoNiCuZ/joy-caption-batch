# Joytag Caption - Batch
This tool utilizes the Joytag Caption tool (still in Alpha), to caption image files in a batch.

Place all images you wish to caption in the /input directory and run `py batch.py`.

Support for an `--img_dir` argument added by [CambridgeComputing](https://github.com/CambridgeComputing) It lets you specify a directory other than ./input. If no arguments are provided by the user, the script still defaults to the ./input subdirectory.

Pre-Alpha version supports `LOW_VRAM_MODE=true`. This will use a [llama3-8b-bnb-4bit quantized version from unsloth](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit)

## Update - 2024-12-08
- Added options for recursive file searches in the input folder.
- Added options for output caption file extension.

## Update - 2024-11-15
Batch Processing value added. You can now edit the script and choose how many files to process at once.
- With JoyCaption Alpha 2, on a 3090, you can use a batch count of 8. 21 images went from 3m to 45s.
- With JoyCaption Prealpha, on a 3090, you can use a batch count of 6. 21 images went from 2m30s to 43s.
- With JoyCaption Prealpha with **low_vram**, on a 3090, you can use a batch count of 16. 21 images went from 3m6s to 47s.

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

- Standard Mode requires ~22gb VRAM.
- Low VRAM Mode requires ~10gb VRAM.
- Low VRAM Mode only works in pre-alpha version.


# Example Pre-Alpha
![put_images_here](https://github.com/user-attachments/assets/a24251e5-6df6-44d4-a231-b74da9fcd8ca)
> This image is a vibrant, detailed digital painting depicting a majestic golden dragon standing on a rocky outcrop in a lush, enchanted garden. The dragon, with its wings spread wide, has a regal, confident posture. Its scales are a shimmering gold, and its wings are a gradient of pink and purple hues, giving them a delicate, almost translucent appearance. The dragon's eyes are red and piercing, and its mouth is open, revealing sharp teeth. In the background, a castle with tall, blue turrets emerges from the mist, adding an air of mystery and fantasy. The castle is partially obscured by trees and foliage, enhancing the sense of a magical, hidden realm. Surrounding the dragon are vibrant flowers in various stages of bloom, including pink and white lilies, blue hydrangeas, and yellow daisies, all adding to the enchanting atmosphere. The garden is lush and verdant, with green leaves and grasses, and the ground is a winding stone pathway that leads towards the castle. The sky is a brilliant blue, dotted with fluffy white clouds, and the light is soft and diffused, creating a dreamlike quality to the entire scene. The overall style is highly detailed and realistic, with a touch of fantasy and whimsy.

![Man](https://github.com/user-attachments/assets/627fe206-fce3-43aa-8ede-b203dfa7f199)
![Woman](https://github.com/user-attachments/assets/6a22d1b5-8a28-42ea-a330-2886ab009aaf)

# Example Alpha 2
![Universpheres](https://github.com/user-attachments/assets/ebb14729-9f96-4123-a719-88eb15d0c98c)



`Write a short description of the image.`
> Two glass spheres, one large and one small, float on a purple cloth, filled with glowing, colorful orbs and intricate, bioluminescent tree branches. The background is a starry night sky.

`Please provide a detailed description of the image.`
> This is a digital artwork depicting two large, spherical glass orbs resting on a dark purple, textured fabric. The orbs are filled with intricate, swirling patterns and vibrant, glowing elements resembling bioluminescent plants and stars. The background is a deep black with scattered, twinkling stars, enhancing the ethereal, cosmic feel. The glass orbs' interior is a mesmerizing mix of blues, purples, oranges, and greens, creating a surreal, otherworldly atmosphere.

`Write a descriptive caption for this image in a formal tone.`
> This is a digital artwork featuring two intricately detailed, translucent spheres placed on a dark purple, textured fabric. The spheres are filled with a complex, swirling pattern of glowing, multicolored orbs and abstract shapes, evoking a sense of cosmic wonder. The background is a starry night sky with a multitude of small, twinkling stars. The spheres' reflective surfaces and vibrant, luminous designs create a mesmerizing contrast against the dark fabric and starry backdrop, blending fantasy and sci-fi elements.

`Write a descriptive caption for this image in a casual tone.`
> Two see-through, round glass orbs sit side by side on a dark purple velvet surface. Inside each one, a wild, colorful scene unfolds with glowing blue, pink, and yellow lights, swirling patterns, and abstract tree-like shapes, all set against a starry night sky backdrop. The whole setup is a trippy, cosmic dream, with a soft, dreamy vibe.

`Write a stable diffusion prompt for this image.`
> A pair of glass orbs, one large and one small, sitting on a purple cloth background. The orbs are filled with a swirling, colorful, abstract pattern that resembles a galaxy or nebula, with blues, purples, oranges, and yellows. The background is a dark, starry night sky with small, twinkling stars. The orbs are highly reflective, showing the intricate details of the pattern inside them. The lighting is soft and even, with no harsh shadows. The overall effect is ethereal and otherworldly. The orbs are placed side by side, with the small one in front of the large one. The purple cloth has a velvety texture and is slightly wrinkled. The orbs are the main focus of the image, with the background providing a sense of depth and context.

`Write a MidJourney prompt for this image.`
> photorealistic, two transparent glass spheres, each containing a surreal, glowing, abstract scene, vibrant colors, black background with scattered stars, intricate designs inside spheres, purple velvet cloth beneath, ethereal, dreamy, cosmic, digital art, sci-fi, fantasy, glowing orbs, intricate patterns, surrealism, dreamlike atmosphere, detailed textures, high contrast, vivid colors, cosmic theme, surreal landscape, abstract art, glowing elements, ethereal beauty, surrealism, sci-fi art, cosmic art, glowing elements, detailed textures, high contrast, vivid colors

`Write a list of Booru tags for this image.`
> digital art, surreal, abstract, space, stars, glowing orbs, blue orbs, purple orbs, yellow orbs, black background, night sky, cosmic, sci-fi, fantasy, glowing, ethereal, dreamy, surrealism, floating orbs, dreamlike, surreal landscape, cosmic background, glowing elements, dark background, celestial, glowing objects, abstract art, digital medium, space-themed, surreal scenery, glowing colors, glowing effects, surreal environment, cosmic scenery, surreal elements, abstract art, glowing orbs, glowing particles, glowing elements, glowing orbs, glowing particles, cosmic scene, surreal art, surreal elements, glowing orbs, glowing particles, glowing elements, cosmic scene, surreal art, surreal elements, glowing orbs, glowing particles, glowing elements, cosmic scene, surreal art, surreal elements, glowing orbs, glowing particles, glowing elements, cosmic scene, surreal art, surreal elements, glowing orbs, glowing particles, glowing elements, cosmic scene, surreal art, surreal elements, glowing orbs, glowing particles, glowing elements, cosmic scene, surreal art, surreal elements, glowing orbs, glowing particles, glowing elements, cosmic scene, surreal art, surreal elements, glowing orbs, glowing particles, glowing elements, cosmic scene, surreal art, surreal elements, glowing orbs, glowing particles, glowing elements, cosmic scene, surreal art, surreal elements, glowing orbs, glowing particles, glowing elements, cosmic scene, surreal art, surreal elements, glowing orbs, glowing particles, glowing elements, cosmic scene, surreal art

Note: This is not stable in Alpha 2. It needs a repeat penalty to stop it from looping

`Write a list of Booru-like tags for this image.`
> digital art, surreal, abstract, space, stars, cosmic, glowing orbs, blue orbs, purple orbs, yellow orbs, black background, night sky, sci-fi, fantasy, dreamy, ethereal, surrealism, glowing, floating, orbs, abstract shapes, surreal landscape, dreamlike, cosmic scene, glowing orbs, dark background, purple fabric, fabric texture, soft texture, glowing light, abstract art, cosmic art, surrealism, space art, glowing orbs, floating orbs, dreamlike scene, abstract shapes, surreal landscape, cosmic scene, glowing orbs, dark background, purple fabric, fabric texture, soft texture, glowing light, abstract art, cosmic art, surrealism, space art, glowing orbs, floating orbs, dreamlike scene, abstract shapes, surreal landscape, cosmic scene, glowing orbs, dark background, purple fabric, fabric texture, soft texture, glowing light, abstract art, cosmic art, surrealism, space art, glowing orbs, floating orbs, dreamlike scene, abstract shapes, surreal landscape, cosmic scene, glowing orbs, dark background, purple fabric, fabric texture, soft texture, glowing light, abstract art, cosmic art, surrealism, space art, glowing orbs, floating orbs, dreamlike scene, abstract shapes, surreal landscape, cosmic scene, glowing orbs, dark background, purple fabric, fabric texture, soft texture, glowing light, abstract art, cosmic art, surrealism, space art, glowing orbs, floating orbs, dreamlike scene, abstract

Note: This is not stable in Alpha 2. It needs a repeat penalty to stop it from looping

`Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.`
> This image showcases a pair of intricately detailed, translucent orbs with a cosmic theme. The orbs are positioned on a rich, dark purple fabric, creating a stark contrast that enhances their luminous quality. The orbs feature a complex network of glowing, ethereal patterns, reminiscent of neural networks or circuitry, with hues of blue, orange, and yellow that evoke a sense of otherworldliness and wonder. The composition is balanced, with the larger orb slightly offset to the right, drawing the viewer's eye across the scene. The use of color is particularly striking; the vibrant, swirling patterns inside the orbs are juxtaposed against the deep, starry background, suggesting a universe within a universe. The lighting is soft and diffused, highlighting the translucency of the orbs and the delicate textures within. The digital effects employed create a sense of depth and dimensionality, with the orbs appearing almost three-dimensional. This image could be associated with the digital surrealism movement, blending elements of fantasy and science fiction with hyper-realistic rendering techniques. The overall effect is mesmerizing, inviting viewers to ponder the mysteries of the cosmos and the potential for life within it.

`Write a caption for this image as though it were a product listing.`
> Discover the mesmerizing "Galactic Bloom" glass sphere set, featuring two intricately designed orbs. Each sphere showcases a stunning, cosmic-inspired pattern with swirling, ethereal blue and purple hues, accented by vibrant orange and yellow highlights. The glass surfaces are smooth and reflective, capturing light beautifully. The spheres rest on a rich, deep purple velvet cloth, enhancing their otherworldly appearance. Perfect for adding a touch of cosmic elegance to any space.

`Write a caption for this image as if it were being used for a social media post.`
> Check out these stunning, glowing orbs! 🌌✨ They look like they're straight out of a sci-fi movie, with intricate, neon-colored patterns and bubbles inside. The larger orb on the right is especially mesmerizing, with vibrant blues, oranges, and purples swirling together. They're sitting on a rich, deep purple fabric that adds an extra layer of magic to the scene. The background is a dreamy, starry night sky, making these orbs feel like they're floating in space. Perfect for anyone who loves a touch of the surreal and the cosmic! 🌠🌌 #SciFiArt #GlowingOrbs #StarryNight #CosmicMagic

## Extra Options
The following can be added to the prompt to guide the direction

- If there is a person/character in the image you must refer to them as {name}.
- Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).
- Include information about lighting.
- Include information about camera angle.
- Include information about whether there is a watermark or not.
- Include information about whether there are JPEG artifacts or not.
- If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.
- Do NOT include anything sexual; keep it PG.
- Do NOT mention the image's resolution.
- You MUST include information about the subjective aesthetic quality of the image from low to very high.
- Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.
- Do NOT mention any text that is in the image.
- Specify the depth of field and whether the background is in focus or blurred.
- If applicable, mention the likely use of artificial or natural lighting sources.
- Do NOT use any ambiguous language.
- Include whether the image is sfw, suggestive, or nsfw.
- ONLY describe the most important elements of the image.

# Run the original online
Original app and source on huggingface: [https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha](https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha)

Alpha 2 script is based on: [https://github.com/fpgaminer/joycaption/blob/main/scripts/batch-caption.py](https://github.com/fpgaminer/joycaption/blob/main/scripts/batch-caption.py)

# Known issues
## Not using the right GPU?
You may need to set the CUDA device to GPU 0 by adding `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` near the start of the code the code.

## Too slow?
You may want to run the model in Low VRAM mode: Set `LOW_VRAM_MODE=true` in batch.py
