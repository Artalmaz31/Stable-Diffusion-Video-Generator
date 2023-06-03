import random
import torch
from stable_diffusion_videos import StableDiffusionWalkPipeline
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models import AutoencoderKL
from diffusers.schedulers import LMSDiscreteScheduler

pipe = StableDiffusionWalkPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=torch.float16,
    safety_checker=None,
    vae=AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to("cuda"),
    scheduler=LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
    )
).to("cuda")

if is_xformers_available():
    pipe.enable_xformers_memory_efficient_attention()

audio_offsets = [0, 5, 10, 20]
fps = 30
num_interpolation_steps = [(b-a) * fps for a, b in zip(audio_offsets, audio_offsets[1:])]

prompts = [
    "A dark alleyway with a single streetlamp illuminating the path ahead",
    "A stormy sky with lightning striking in the distance.",
    "A person walking through a deserted city street."
]

seeds = [random.randint(0, 9e9) for _ in range(len(prompts))]

pipe.walk(
    prompts=prompts,
    seeds=seeds,
    num_interpolation_steps=num_interpolation_steps,
    fps=fps,
    audio_start_sec=audio_offsets[0],
    batch_size=4,
    num_inference_steps=50,
    guidance_scale=15,
    margin=1.0,
    smooth=0.2,
)