import torch
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID


app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


base_model_path = "emilianJR/chilloutmix_NiPrunedFp32Fix"
#base_model_path = "jzli/PerfectWorld-v5"
base_model_path = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
ip_ckpt = "models/ip-adapter-faceid_sd15.bin"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

#pipe = StableDiffusionPipeline.from_pretrained( base_model_path, torch_dtype=torch.float16, variant="fp16")
pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None)

pipe.safety_checker=None
pipe.require_safety_checker=False

pipe = pipe.to("cuda")

ip_model = IPAdapterFaceID(pipe, ip_ckpt, "cuda")

image = cv2.imread("workspace/other/1.png")
faces = app.get(image)
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

#prompt = "photo of a woman in red dress in a garden"
prompt = "8K, best quality, masterpiece, photograph, 1girl"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"
images = ip_model.generate(
    prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=4, width=512, height=768, num_inference_steps=30, seed=-1
)

#image = pipe("8K, best quality, masterpiece, photograph, girl").images[0]

for i in range(4):
    with open("output/real_%s.jpeg"%i, "w") as f:
        images[i].save(f, format="JPEG")
