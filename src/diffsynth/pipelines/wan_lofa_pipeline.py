import os
import json
import argparse
from typing import Any, Optional, Dict, List
from types import MethodType
from time import time
from hydra.utils import instantiate
from safetensors.torch import load_file
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer, Linear
from einops import rearrange

import torch
import tqdm
from ..pipelines import WanVideoPipeline
from ..pipelines.wan_video import model_fn_wan_video

from ..models import ModelManager
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter

def apply_custom_forward(model, custom_fn):
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer) or (hasattr(m, "lora_A") and hasattr(m, "lora_B")):
            m.forward = MethodType(custom_fn, m)

VALID_LAYERS = [
    "self_attn_q", "self_attn_k", "self_attn_v", "self_attn_o",
    "cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_o",
    # "ffn_0", "ffn_2"
]



class WanLofaPipeline(WanVideoPipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device, torch_dtype, tokenizer_path)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.model_names = ['text_encoder', 'dit', 'vae', 'image_encoder', 'motion_controller', 'vace']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.use_unified_sequence_parallel = False

    def add_lofa(self, cfg, lofa_path, wan_dit_path):
        self.lofa = instantiate(cfg.stage2, _recursive_=False)
        state = torch.load(lofa_path, map_location="cpu")
        load_result = self.lofa.load_state_dict(state, strict=False)
        if load_result.missing_keys:
            print(f"[Warn] Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"[Warn] Unexpected keys: {load_result.unexpected_keys}")
        self.lofa.to(device=self.device, dtype=self.torch_dtype)
        self.lofa.eval()
        self.wan_params = load_file(wan_dit_path)
        self.wan_params = {k: v.to(device=self.device, dtype=self.torch_dtype) for k, v in self.wan_params.items()}
        self.inject_lora()
    
    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanLofaPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True
        return pipe

    @staticmethod
    def add_lora_to_model(model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model

    @staticmethod
    def add_lora_names(model):
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer) or (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                module._name_in_model  = name
    
    def inject_lora(self):
        self.dit = self.add_lora_to_model(
            self.dit,
            target_modules=["q", "k", "v", "o"],
            lora_rank=32
        )
        self.add_lora_names(self.dit)


    @staticmethod
    def update_lora_weights(lora_A, lora_B, alpha, layer_indices, layer_types):
        batch_size = 1
        lora_A = rearrange(lora_A, "(B L) r c -> B L c r", B=batch_size)
        lora_B = rearrange(lora_B, "(B L) c r -> B L r c", B=batch_size)
        alpha = rearrange(alpha, "(B L) C -> B (L C)", B=batch_size)
        layer_indices = rearrange(layer_indices, "(B L) -> B L", B=batch_size)[0]
        layer_types = rearrange(layer_types, "(B L) -> B L", B=batch_size)[0]
        state_dict = {}
        for i, (depth, layer_type) in enumerate(zip(
            layer_indices, layer_types
        )):
            module_name = VALID_LAYERS[layer_type.cpu().item()]
            block_name= f"blocks.{depth}.{module_name[:-2]}.{module_name[-1]}"
            state_dict[block_name] = dict(lora_A = lora_A[:, i, :, :], lora_B = lora_B[:, i, :, :], alpha=alpha[:, i])
        
        return state_dict
 
    def prepare_lofa_context(self, prompt_emb, input_prior=None):
        if prompt_emb.dim() == 2:
            prompt_emb = prompt_emb.unsqueeze(0)
        layer_indices = []
        layer_types = []
        input_params = []
        input_priors = []
        for depth in range(30):
            for layer_type, module_name in enumerate(VALID_LAYERS):
                layer_indices.append(depth)
                layer_types.append(layer_type)
                block_name= f"blocks.{depth}.{module_name[:-2]}.{module_name[-1]}.weight"
                input_params.append(self.wan_params[block_name])
                if input_prior is not None:
                    reponse_map = input_prior[f"blocks.{depth}.{module_name[:-2]}.{module_name[-1]}"]
                    reponse_map = torch.from_numpy(reponse_map).to(self.device, dtype=self.torch_dtype)
                    input_priors.append(reponse_map)
        if input_prior is not None:
            input_priors = torch.stack(input_priors, dim=0).to(self.device, dtype=self.torch_dtype)
        input_params = torch.stack(input_params, dim=0).to(self.device, dtype=self.torch_dtype)
        layer_indices = torch.tensor(layer_indices, device=prompt_emb.device).long()
        layer_types = torch.tensor(layer_types, device=prompt_emb.device).long()
        prompt_emb = prompt_emb.repeat(len(layer_indices), 1, 1)
        context = {
            "W":input_params,
            "text_emb": prompt_emb,
            "depth_id": layer_indices,
            "type_id": layer_types
        }
        if input_prior is not None:
            context["input_prior"] = input_priors
        return context

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        end_image=None,
        input_video=None,
        response_map=None,
        control_video=None,
        reference_image=None,
        vace_video=None,
        vace_video_mask=None,
        vace_reference_image=None,
        vace_scale=1.0,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        motion_bucket_id=None,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        lora_scale=1.0,
    ):
        height, width = self.check_resize_height_width(height, width)

        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 == 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Initialize noise
        noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(['vae'])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise

        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
        
        lofa_context = self.prepare_lofa_context(prompt_emb_posi["context"], input_prior=response_map)

        A, B, alpha = self.lofa(**lofa_context)
        lora_param_dict = self.update_lora_weights(A, B, alpha, layer_indices=lofa_context["depth_id"], layer_types=lofa_context["type_id"]) 
        # print(f"fomatting prediction time: {time() - start_time}")
        # start_time = time()
        def custom_lora_linear_forward(self: Linear, x: torch.Tensor, *args: Any, **kwargs: Any)  -> torch.Tensor:
            self._check_forward_args(x, *args, **kwargs)  
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            lora_A, lora_B, alpha = lora_param_dict[self._name_in_model]["lora_A"], lora_param_dict[self._name_in_model]["lora_B"], lora_param_dict[self._name_in_model]["alpha"]

            x = self._cast_input_dtype(x, lora_A.dtype)
            delta_x = torch.bmm(torch.bmm(x, lora_A), lora_B) * alpha[:, None, None] * lora_scale
            result += delta_x
            result = result.to(torch_result_dtype)

            return result
        apply_custom_forward(self.dit, custom_lora_linear_forward)


        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, end_image, num_frames, height, width, **tiler_kwargs)
        else:
            image_emb = {}
            
        # Reference image
        reference_image_kwargs = self.prepare_reference_image(reference_image, height, width)
            
        # ControlNet
        if control_video is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.prepare_controlnet_kwargs(control_video, num_frames, height, width, **image_emb, **tiler_kwargs)
            
        # Motion Controller
        if self.motion_controller is not None and motion_bucket_id is not None:
            motion_kwargs = self.prepare_motion_bucket_id(motion_bucket_id)
        else:
            motion_kwargs = {}
            
        # Extra input
        extra_input = self.prepare_extra_input(latents)
        
        # VACE
        latents, vace_kwargs = self.prepare_vace_kwargs(
            latents, vace_video, vace_video_mask, vace_reference_image, vace_scale,
            height=height, width=width, num_frames=num_frames, seed=seed, rand_device=rand_device, **tiler_kwargs
        )
        
        # TeaCache
        tea_cache_posi = {"tea_cache": None}
        tea_cache_nega = {"tea_cache": None}
        
        # Unified Sequence Parallel
        usp_kwargs = self.prepare_unified_sequence_parallel()

        # Denoise
        self.load_models_to_device(["dit", "motion_controller", "vace"])
        pbar = tqdm.tqdm(total=len(self.scheduler.timesteps), desc="Sampling")
        for progress_id, timestep in enumerate(self.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = model_fn_wan_video(
                self.dit, motion_controller=self.motion_controller, vace=self.vace,
                x=latents, timestep=timestep,
                **prompt_emb_posi, **image_emb, **extra_input,
                **tea_cache_posi, **usp_kwargs, **motion_kwargs, **vace_kwargs, **reference_image_kwargs,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = model_fn_wan_video(
                    self.dit, motion_controller=self.motion_controller, vace=self.vace,
                    x=latents, timestep=timestep,
                    **prompt_emb_nega, **image_emb, **extra_input,
                    **tea_cache_nega, **usp_kwargs, **motion_kwargs, **vace_kwargs, **reference_image_kwargs,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
            pbar.update(1)
        if vace_reference_image is not None:
            latents = latents[:, :, 1:]

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames