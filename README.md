# diffusion_gen

## Tutorial

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