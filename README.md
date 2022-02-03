# Diffusion Gen

An adapation of DiscoDiffusion (https://colab.research.google.com/drive/1sHfRn5Y0YKYKi1k-ifUSBFRNJ8_1sa39#scrollTo=BGBzhk3dpcGO) to run locally, to improve code quality and to speed it up. So far the code was just cleaned up a bit and the lpips network initialization was removed when only an input text is used.

Around 11GB GPU VRAM are needed for the current default settings of `--width` 1280 and `--height` 768. Decreasing the image size is the easiest way to make it fit in smaler GPUs.
With defaults settings it takes 07:46 minutes on an RTX 2080TI, 19:01 minutes on a GTX 1080 TI, and 17:01 minutes on a Titan XP to generate images like these:


**The meaning of life**
![meaning(0)_0](https://user-images.githubusercontent.com/19983153/150617587-0b1396bd-339f-4867-8a4a-c15bb75fd71a.png)


**The meaning of life by Picasso**
![meaning(2)_0](https://user-images.githubusercontent.com/19983153/150617599-4ceb2896-9aa1-4497-b7ad-80c488f68938.png)


**The meaning of life by Greg Rutkowski**
![meaning_rutkowski](https://user-images.githubusercontent.com/19983153/150616859-0630e090-d737-4ced-9893-4a2c9937a949.png)

**Consciousness**
![out_image(0)_0](https://user-images.githubusercontent.com/19983153/150617545-1048b160-084c-4854-adc3-6afb13731fdf.png)

*forgot the prompt but it was about pikachu staring at a tumultous sea of blood, adapted from the DiscoDiffusion original notebook*
![Screenshot from 2022-01-21 15-35-09](https://user-images.githubusercontent.com/19983153/150616643-54436dbc-1e38-4127-b0dd-f0097470ae0f.png)


## Setup
If you're using Windows, please also refer to the section below called `Setup for Windows`!

First run `ipython3 diffuse.py` to set everything up and to clone the repositories. IMPORTANT: you need to use ipython instead of python because I was lazy and all git clone etc are run via ipython

At the moment you can only set a single text as a target but this should be improved in the future. Only runs with GPU support atm.

Use it like this:

```
python3 diffuse.py --text "The meaning of life --gpu [Optional: device number of GPU to run this on] --root_path [Optional: path to output folder, default is "out_diffusion" in local dir]
```
If you only have 8 GB VRAM on your GPU, the highest resolution you can use run is 832x512, or 896x448. Set it by adding `--width 832 --height 512` for example. Thanks @Jotunblood for testing! 

you can also set: `--out_name [Optional: set naming in your root_path according to this for better overview]` and `--sharpen_preset [Optional: set it to any of ('Off', 'Faster', 'Fast', 'Slow', 'Very Slow') to modify the sharpening process at the end. Default: Off]`


## Setup for Windows
See https://github.com/NotNANtoN/diffusion_gen/issues/1, the instructions from @JotunBlood are adopted here.

Instructions:
- Install Anaconda
- Create and activate a new environment (don't use base)
- Install pytorch via their web code, using pip (not conda)
- Install iPython
- Add the forge channel to anaconda
- conda config --add channels conda-forge
- Install dependency packages using conda (for those available), otherwise use pip. Packages of relevance: OpenCV, pandas, timm, lpips, requests, pytorch-lightning, and omegaconf. There might be one or two others.
- Run ipython diffuse.py
- If it goes all the way, congrats. If you hit the SSL errors, open diffuse.py and add the following lines to the top of diffuse.py to the top (I did it around line 7.):
 ```
 import ssl
 ssl._create_default_https_context = ssl._create_unverified_context
 ```
- If you get Frame Prompt: [''] and a failed output, make sure you're using python3 to run diffuse.py and not iPython :)
- If you get a CUDA out of memory warning, pass a lower res like --width 720 --height 480 when you run

## Tutorial (copypasta from old colab notebook)

### **Diffusion settings**
 ---
 
 This section is outdated as of v2
 
 Setting | Description | Default
 --- | --- | ---
 **Your vision:**
 `text_prompts` | A description of what you'd like the machine to generate. Think of it like writing the caption below your image on a website. | N/A
 `image_prompts` | Think of these images more as a description of their contents. | N/A
 **Image quality:**
 `clip_guidance_scale`  | Controls how much the image should look like the prompt. | 1000
 `tv_scale` |  Controls the smoothness of the final output. | 150
 `range_scale` |  Controls how far out of range RGB values are allowed to be. | 150
 `sat_scale` | Controls how much saturation is allowed. From nshepperd's JAX notebook. | 0
 `cutn` | Controls how many crops to take from the image. | 16
 `cutn_batches` | Accumulate CLIP gradient from multiple batches of cuts  | 2
 **Init settings:**
 `init_image` |   URL or local path | None
 `init_scale` |  This enhances the effect of the init image, a good value is 1000 | 0
 `skip_steps Controls the starting point along the diffusion timesteps | 0
 `perlin_init` |  Option to start with random perlin noise | False
 `perlin_mode` |  ('gray', 'color') | 'mixed'
 **Advanced:**
 `skip_augs` |Controls whether to skip torchvision augmentations | False
 `randomize_class` |Controls whether the imagenet class is randomly changed each iteration | True
 `clip_denoised` |Determines whether CLIP discriminates a noisy or denoised image | False
 `clamp_grad` |Experimental: Using adaptive clip grad in the cond_fn | True
 `seed`  | Choose a random seed and print it at end of run for reproduction | random_seed
 `fuzzy_prompt` | Controls whether to add multiple noisy prompts to the prompt losses | False
 `rand_mag` |Controls the magnitude of the random noise | 0.1
 `eta` | DDIM hyperparameter | 0.5
