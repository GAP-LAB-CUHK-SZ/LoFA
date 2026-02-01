import argparse
import os
import json
from collections import defaultdict
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from hydra import initialize, compose
from hydra.utils import instantiate
from accelerate.utils import set_seed

import torch
from safetensors.torch import load_file

from diffsynth import ModelManager
from diffsynth.prompters import WanPrompter
from lofa.dataset import VALID_LAYERS

def load_text_encoder():
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    text_encoder, tokenizer_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
    prompter = WanPrompter(tokenizer_path=None)
    prompter.fetch_models(text_encoder)
    prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
    prompter.text_encoder.eval()
    prompter.text_encoder.to("cuda")

    return prompter

def prepare_lofa_context(wan_params, prompt_emb, device, dtype):
    if prompt_emb.dim() == 2:
        prompt_emb = prompt_emb.unsqueeze(0)
    layer_indices = []
    layer_types = []
    input_params = []
    for depth in range(30):
        for layer_type, module_name in enumerate(VALID_LAYERS):
            layer_indices.append(depth)
            layer_types.append(layer_type)
            block_name= f"blocks.{depth}.{module_name[:-2]}.{module_name[-1]}.weight"
            input_params.append(wan_params[block_name])

    input_params = torch.stack(input_params, dim=0).to(device, dtype=dtype)

    layer_indices = torch.tensor(layer_indices, device=prompt_emb.device).long()
    layer_types = torch.tensor(layer_types, device=prompt_emb.device).long()
    prompt_emb = prompt_emb.repeat(len(layer_indices), 1, 1)
    context = {
        "W":input_params,
        "text_emb": prompt_emb,
        "depth_id": layer_indices,
        "type_id": layer_types
    }
    return context

def main():
    model_path = "checkpoints/test_release"
    with initialize(version_base=None, config_path=model_path):
        cfg = compose(config_name="config")
    device = "cuda"
    dtype = (torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16 if cfg.mixed_precision == "fp16" else torch.float32)
    # Load model
    model = instantiate(cfg.stage1).to(dtype).to("cuda")
    model.load_state_dict(torch.load(os.path.join(model_path, "stage1_preview.pth"), map_location="cpu"), strict=False)
    model.eval()
    prompter = load_text_encoder()

    # load wan params
    wan_params = load_file("models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors")
    wan_params = {k: v.to(device=device, dtype=dtype) for k, v in wan_params.items()}

    prompt = ""
    with torch.no_grad():
        prompt_emb = prompter.encode_prompt(prompt, positive=True, device=device)
        context = prepare_lofa_context(wan_params, prompt_emb, device, dtype)
        hidden_states, logits, pred_prob = model.forward_features(**context)
    result_dict = {}
    for j, (depth, layer_idx) in enumerate(zip(context['depth_id'], context['type_id'])):
        module_name = VALID_LAYERS[layer_idx.item()]
        block_name = f"blocks.{depth}.{module_name[:-2]}.{module_name[-1]}"
        result_dict[block_name] = hidden_states[j].cpu().half().numpy()
    np.savez_compressed("stage1_feat.npz", **result_dict)

if __name__ == "__main__":
    main()
