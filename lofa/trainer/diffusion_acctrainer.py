import os
import gc
import logging
import random
from time import time
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from hydra.utils import instantiate
from types import MethodType
from typing import Any, Dict, List, Mapping, Optional, Sequence
from einops import rearrange
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from peft.tuners.lora.layer import Linear
from safetensors.torch import load_file

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed

from diffusers.optimization import get_scheduler

from lofa.train_utils import is_compiled_module, grad_norm_of_loss
from lofa.dataset import collate_diff
from lofa.losses import get_loss_fn

from diffsynth import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline

INIT_A = torch.zeros([1, 1536, 32], dtype=torch.bfloat16)
INIT_B = torch.zeros([1, 32, 1536], dtype=torch.bfloat16)
LORA_STATE_DICT = {}
def update_globals(new_A, new_B):
    global INIT_A, INIT_B  
    INIT_A = new_A
    INIT_B = new_B

def iter_lora_layers(model):
    layers = {}
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer) or (hasattr(m, "lora_A") and hasattr(m, "lora_B")):
            layers[name] = m
    return layers

def apply_custom_forward(model, custom_fn):
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer) or (hasattr(m, "lora_A") and hasattr(m, "lora_B")):
            m.forward = MethodType(custom_fn, m)

def add_lora_names(model):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) or (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            module._name_in_model  = name
    

class DiffTrainer:
    def __init__(
        self,
        save_path: str,
        exp_name: str,
        data: Dict[str, Any],
        model: Dict[str, Any],
        checkpoint: Dict[str, Any],
        loss: Dict[str, Any],
        wan: Dict[str, Any],
        max_steps: int,
        mixed_precision: str = "bf16",
        seed_value: int = 123,
        optim: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        batch_size: int = 1,
        report_to: str = "tensorboard",
        scale_factor: float = 1.0,
        diffusion_loss_weight: float = 1.0,
        **kwargs
    ):
        self.data_conf = data
        self.model_conf = model
        self.optim_conf = optim
        self.checkpoint_conf = checkpoint
        self.loss_conf = loss
        self.exp_name = exp_name
        self.save_path = save_path
        self.wan_conf = wan

        self.accum_steps = accum_steps
        self.batch_size = batch_size
        self.precision = mixed_precision
        self.scale_factor = scale_factor
        self.diffusion_loss_weight = diffusion_loss_weight
        if self.precision == "fp16":
            self.dtype = torch.float16
        elif self.precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.max_steps = max_steps
        self.seed_value = seed_value

        logging_dir = Path(self.save_path, "logging")
        accelerator_project_config = ProjectConfiguration(project_dir=self.save_path, logging_dir=logging_dir)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.accum_steps,
            mixed_precision=self.precision,
            log_with=report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger = get_logger(__name__)
        self.logger.info(self.accelerator.state, main_process_only=False)
        set_seed(self.seed_value)

        self.model = instantiate(self.model_conf, _recursive_=False)
        self._init_wan()
        from lofa.dataset import WanHyperDiffDataset
        # from hyper_diff_modulator.dataset import WanHyperDiffDataset_old as WanHyperDiffDataset
        self.train_dataset = WanHyperDiffDataset(**self.data_conf)
        # self.train_dataset = instantiate(self.data_conf, _recursive_=False)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.data_conf.num_workers,
            collate_fn=collate_diff,
            # pin_memory=True,
            # persistent_workers=True 
        )

        self.model.train()
        self.model.requires_grad_(True)
        self.model.to(self.accelerator.device, dtype=self.dtype)

        self.accelerator.register_save_state_pre_hook(self.save_model_hook)
        self.accelerator.register_load_state_pre_hook(self.load_model_hook)

        self.params_to_optimize = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.grad_check_params = [p for p in self.model.blocks[-1].parameters()]

        self.optimizer = torch.optim.AdamW(
            self.params_to_optimize,
            lr=self.optim_conf.learning_rate,
            betas=(self.optim_conf.adam_beta1, self.optim_conf.adam_beta2),
            weight_decay=self.optim_conf.adam_weight_decay,
            eps=self.optim_conf.adam_epsilon,
        )

        self.lr_scheduler = get_scheduler(
            self.optim_conf.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.optim_conf.lr_warmup_steps,
            num_training_steps=self.max_steps
        )

        self.global_step = 0
        if self.checkpoint_conf.resume_from_checkpoint:
            if self.checkpoint_conf.resume_from_checkpoint != "latest":
                path = self.checkpoint_conf.resume_from_checkpoint
            else:
                dirs = os.listdir(self.save_path)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
                path = os.path.join(self.save_path, path)
            
            self.accelerator.print(f"Resuming from checkpoint {path}")
            ckpt = torch.load(os.path.join(path, "model.pth"))
            self.model.load_state_dict(ckpt)
            self.global_step = int(path.split("-")[-1])

        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler, self.pipe = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler, self.pipe
        )

        # Afterwards we recalculate our number of training epochs
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(self.exp_name)

        self.total_batch_size = self.batch_size * self.accelerator.num_processes * self.accum_steps * len(self.data_conf.valid_layers) * self.data_conf.n_layers
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Instantaneous batch size per device = {self.batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.accum_steps}")
        self.logger.info(f"  Total optimization steps = {self.max_steps}")

        self.global_step = 0
        if self.checkpoint_conf.resume_from_checkpoint:
            if self.checkpoint_conf.resume_from_checkpoint != "latest":
                path = self.checkpoint_conf.resume_from_checkpoint
            else:
                dirs = os.listdir(self.save_path)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None
                path = os.path.join(self.save_path, path)
            
            self.accelerator.print(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(path)
            self.global_step = int(path.split("-")[1])

        self.valid_layers = self.train_dataset.valid_layers
        self.n_layers = self.train_dataset.n_layers
        self.wan_params = load_file(self.wan_conf.model_path)
        self.wan_params = {k: v.to(self.dtype) for k, v in self.wan_params.items()}

    def _init_wan(self):
        # Load models
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(self.wan_conf.model_path):
            model_manager.load_models([self.wan_conf.model_path])
        else:
            dit_path = self.wan_conf.model_path.split(",")
            model_manager.load_models([dit_path])

        self.pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16)

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        # Add LoRA to the base models
        lora_base_model = "dit"
        model = self.add_lora_to_model(
            getattr(self.pipe, lora_base_model),
            target_modules=self.wan_conf.target_modules.split(","),
            lora_rank=self.model_conf.lora_rank
        )
        setattr(self.pipe, lora_base_model, model)
        self.pipe.requires_grad_(False)
        self.pipe.denoising_model().train()
        self.pipe.to(self.accelerator.device, dtype=torch.bfloat16)

        # Store other configs
        self.use_gradient_checkpointing = self.wan_conf.use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = self.wan_conf.use_gradient_checkpointing_offload
        self.__orig_forward = next(
            (m for name, m in model.named_modules()
            if isinstance(m, LoraLayer) or (hasattr(m, "lora_A") and hasattr(m, "lora_B")))
        ).forward
        
        add_lora_names(self.pipe.dit)        
        
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model  
    
    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    def save_model_hook(self, models, weights, output_dir):
        state = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            torch.save(state, os.path.join(output_dir, f"model.pth"))
        weights.clear()
    
    def load_model_hook(self, models, input_dir):
        model = models[0]
        state = torch.load(os.path.join(input_dir, "model.pth"), map_location="cpu")
        model.load_state_dict(state, strict=False)

    def update_lora_weights(self, lora_A, lora_B, alpha, layer_indices, layer_types):
        lora_A = rearrange(lora_A, "(B L) r c -> B L c r", B=self.batch_size)
        lora_B = rearrange(lora_B, "(B L) c r -> B L r c", B=self.batch_size)
        alpha = rearrange(alpha, "(B L) C -> B (L C)", B=self.batch_size)
        layer_indices = rearrange(layer_indices, "(B L) -> B L", B=self.batch_size)[0]
        layer_types = rearrange(layer_types, "(B L) -> B L", B=self.batch_size)[0]
        state_dict = {}
        for i, (depth, layer_type) in enumerate(zip(
            layer_indices, layer_types
        )):
            module_name = self.train_dataset.valid_layers[layer_type.cpu().item()]
            block_name= f"blocks.{depth}.{module_name[:-2]}.{module_name[-1]}"
            state_dict[block_name] = dict(lora_A = lora_A[:, i, :, :], lora_B = lora_B[:, i, :, :], alpha=alpha[:, i])
        
        return state_dict


    def run(self):
        progress_bar = tqdm(
            range(0, self.max_steps),
            initial=self.global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )
        ### prepare loss dict
        all_loss_fn = {}
        gathered_loss_dict = {}
        train_loss = 0.0
        loss_dict = {}
        if self.loss_conf is not None:
            for loss_name, loss_conf in self.loss_conf.items():
                all_loss_fn[loss_name] = get_loss_fn(loss_conf.type)

        for _, batch in enumerate(self.train_dataloader):
            if self.global_step >= self.max_steps:
                break

            with self.accelerator.accumulate(self.model):
                # start_time = time()
                batch, latents, context, input_params = self._process_batch(batch)
                with self.accelerator.autocast():
                    pred = self.model(
                        input_params,
                        batch["layer_indices"],
                        batch["layer_type"],
                        batch["text_emb"],
                        batch["input_prior"] if "input_prior" in batch else None
                    )
                # print(f"hypermodel forward time: {time() - start_time}")
                # start_time = time()
                target_A = batch["lora_A"].float()
                target_B = batch["lora_B"].float()
                A, B, alpha = pred
                # if self.model_conf.factorized:
                #     A = (A * self.std_A + self.mean_A) / self.scale_factor
                #     B = (B * self.std_B + self.mean_B) / self.scale_factor
                loss = 0.0
                for key, loss_fn in all_loss_fn.items():
                    loss_value = loss_fn([A, B], [target_A, target_B], **self.loss_conf[key])
                    loss += loss_value * self.loss_conf[key].loss_weight
                    loss_dict[key] = loss_value.detach()

                g_recon = grad_norm_of_loss(loss, self.grad_check_params, retain_graph=True)
                
                ##################                dit forward  ###################### 
                with self.accelerator.autocast(): 
                    lora_param_dict = self.update_lora_weights(A, B, alpha, layer_indices=batch["layer_indices"], layer_types=batch["layer_type"]) 
                    # print(f"fomatting prediction time: {time() - start_time}")
                    # start_time = time()
                    def custom_lora_linear_forward(self: Linear, x: torch.Tensor, *args: Any, **kwargs: Any)  -> torch.Tensor:
                        self._check_forward_args(x, *args, **kwargs)  
                        result = self.base_layer(x, *args, **kwargs)
                        torch_result_dtype = result.dtype
                        lora_A, lora_B, alpha = lora_param_dict[self._name_in_model]["lora_A"], lora_param_dict[self._name_in_model]["lora_B"], lora_param_dict[self._name_in_model]["alpha"]

                        x = self._cast_input_dtype(x, lora_A.dtype)
                        delta_x = torch.bmm(torch.bmm(x, lora_A), lora_B) * alpha[:, None, None] 
                        result += delta_x
                        result = result.to(torch_result_dtype)

                        return result
                    apply_custom_forward(self.pipe.dit, custom_lora_linear_forward)
                diffusion_loss = self._forward_dit(latents, context)
                loss_dict["diffusion_loss"] = diffusion_loss.detach()
                diffusion_loss = diffusion_loss * self.diffusion_loss_weight
                g_diff = grad_norm_of_loss(diffusion_loss, self.grad_check_params, retain_graph=True)
                g_ratio = g_recon / (g_diff + 1e-12)
                # print(f"diffusion time: {time() - start_time}")
                # start_time = time()
                loss += diffusion_loss.float()

                for key, loss_value in loss_dict.items():
                    if key not in gathered_loss_dict:
                        gathered_loss_dict["loss/" + key] = 0.0
                    gathered_loss = self.accelerator.gather(loss_value.repeat(self.batch_size)).mean().item() / self.accum_steps
                    gathered_loss_dict["loss/" + key] = gathered_loss
                train_loss += self.accelerator.gather(loss.repeat(self.batch_size)).mean().item() / self.accum_steps
                # Backpropagate
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    # grad_dict = {}
                    grad_norm = self.accelerator.clip_grad_norm_(self.params_to_optimize, self.optim_conf.grad_norm)
                    # if self.optim_conf.grad_log_keys:
                    #     for name, param in self.unwrap_model(self.model).named_parameters():
                    #         if param.grad is not None and name in self.optim_conf.grad_log_keys:
                    #             param_norm = param.grad.detach().norm(2).item()
                    #             grad_dict["grad/" + name + '_grad'] = param_norm
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                # print(f"bp time: {time() - start_time}")
                # start_time = time()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                lr = self.lr_scheduler.get_last_lr()[0]
                log_dict = {
                    "train_loss": train_loss, 
                    "lr": lr,
                    "grad_clip": grad_norm.item(),
                    "g_ratio": g_ratio
                }
                log_dict.update(gathered_loss_dict)
                # log_dict.update(grad_dict)
                self.accelerator.log(log_dict, step=self.global_step)

                train_loss = 0.0
                gathered_loss_dict = {}
                self.global_step += 1

                if self.global_step % self.checkpoint_conf.save_freq == 0 or self.global_step == self.max_steps:
                    if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
                        save_path = os.path.join(self.save_path, f"checkpoint-{self.global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        self.accelerator.save_state(save_path)
                        self.logger.info(f"Saved state to {save_path}")
                    self.accelerator.wait_for_everyone()
                logs = {"step_loss": loss.detach().item(), "lr": lr}
            # logs.update(grad_dict or {})
                progress_bar.set_postfix(**logs)
        del self.train_dataloader
        gc.collect()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
                save_path = os.path.join(self.save_path, f"checkpoint-{self.global_step}")
                os.makedirs(save_path, exist_ok=True)
                self.accelerator.save_state(save_path)
                self.logger.info(f"Saved state to {save_path}")
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        torch.cuda.empty_cache()
    
    def _forward_dit(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.accelerator.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, context=context,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        return loss

    def _process_batch(self, batch):

        sampled_layer_indices = batch["layer_depth"].to(self.accelerator.device, non_blocking=True).long()
        sampled_layer_types = batch["layer_type"].to(self.accelerator.device, non_blocking=True).long()
        sampled_lora_A = batch["lora_A"].to(self.accelerator.device, dtype=self.dtype, non_blocking=True)
        sampled_lora_B = batch["lora_B"].to(self.accelerator.device, dtype=self.dtype, non_blocking=True)
        text_emb = batch["text_emb"].to(self.accelerator.device, dtype=self.dtype, non_blocking=True)
        latent = batch["latent"].to(self.accelerator.device, dtype=self.dtype, non_blocking=True)
        context = batch["context"].to(self.accelerator.device, dtype=self.dtype, non_blocking=True)
        input_prior = batch["input_prior"].to(self.accelerator.device, dtype=self.dtype, non_blocking=True) if batch["input_prior"] is not None else None

        input_params = []
        for i, (depth, layer_type) in enumerate(zip(
            sampled_layer_indices, sampled_layer_types
        )):
            module_name = self.train_dataset.valid_layers[layer_type.cpu().item()]
            block_name= f"blocks.{depth}.{module_name[:-2]}.{module_name[-1]}.weight"
            input_params.append(self.wan_params[block_name])
        input_params = torch.stack(input_params).to(self.accelerator.device, non_blocking=True)


        return dict(layer_indices=sampled_layer_indices, layer_type=sampled_layer_types, lora_A=sampled_lora_A, lora_B=sampled_lora_B, text_emb=text_emb, input_prior=input_prior), latent, context, input_params

    
