import os
import numpy as np
from hydra import compose, initialize
from hydra.utils import instantiate
import torch
from diffsynth import ModelManager, save_video

from diffsynth.pipelines.wan_lofa_pipeline import WanLofaPipeline

model_path = "checkpoints/test_release"
with initialize(version_base=None, config_path=model_path):
    cfg = compose(config_name="config")

dtype = (
    torch.bfloat16
    if cfg.mixed_precision == "bf16"
    else torch.float16
    if cfg.mixed_precision == "fp16"
    else torch.float32
)

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    [
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
pipe = WanLofaPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.add_lofa(cfg, os.path.join(model_path, "stage2_preview.pth"), "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

prompt = ""
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，畸形的，毁容的，形态畸形的肢体，静止不动的画面，杂乱的背景，倒着走"

# load response map from npz file
response_map = np.load("stage1_feat.npz")
# Text-to-video
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    response_map=response_map,
    num_inference_steps=50,
    seed=42, tiled=True,
    num_frames=81,
    height=480,
    width=832,
    lora_scale=1.0,
)
save_video(video, "output.mp4", fps=8, quality=5)