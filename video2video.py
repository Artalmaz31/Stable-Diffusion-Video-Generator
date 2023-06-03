import torch
from diffusers import StableDiffusionImg2ImgPipeline
import cv2
from PIL import Image
import numpy

device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    force_download=True,
    resume_download=False,
    safety_checker = None
)
pipe = pipe.to(device)

video_file = "video.mp4"
capture = cv2.VideoCapture(video_file)
prompt = "cinematic style, strong colors, a masterpiece"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(
    filename="output.mp4", fourcc=fourcc, fps=30.0, frameSize=(1920, 1080)
)

while True:
    success, frame = capture.read()
 
    if success:
        generator = torch.Generator(device=device).manual_seed(1024)
        init_image = Image.fromarray(frame).convert("RGB")
        init_image = init_image.resize((1920, 1080))
        image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5, generator=generator).images[0]
        image = numpy.asarray(image)
        image = cv2.resize(image, dsize=(1920, 1080))
        video.write(image)
 
    else:
        break

video.release()