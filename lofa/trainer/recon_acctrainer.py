
import gc
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed

from diffusers.optimization import get_scheduler

from lofa.dataset import WanReconMaskDataset, collate_mask
from lofa.losses import get_loss_fn
from lofa.train_utils import is_compiled_module
from lofa.lora_utils import plot_heatmap


class ReconTrainer:
    def __init__(
        self,
        save_path: str,
        exp_name: str,
        data: Dict[str, Any],
        model: Dict[str, Any],
        checkpoint: Dict[str, Any],
        loss: Dict[str, Any],
        max_steps: int,
        mixed_precision: str = "bf16",
        seed_value: int = 123,
        optim: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        batch_size: int = 1,
        report_to: str = "tensorboard",
        scale_factor: float = 1.0,
        wan: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.data_conf = data
        self.model_conf = model
        self.optim_conf = optim
        self.checkpoint_conf = checkpoint
        self.loss_conf = loss
        self.exp_name = exp_name
        self.save_path = save_path
        self.wan_conf = wan
        self.scale_factor = scale_factor

        self.tau = self.model_conf.get("tau", 0.02)

        self.accum_steps = accum_steps
        self.batch_size = batch_size
        self.precision = mixed_precision
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
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.accum_steps,
            mixed_precision=self.precision,
            log_with=report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
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
        self.train_dataset = WanReconMaskDataset(**self.data_conf)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.data_conf.num_workers,
            collate_fn=collate_mask,
        )

        self.model.train()
        self.model.requires_grad_(True)
        self.model.to(self.accelerator.device, dtype=self.dtype)

        self.accelerator.register_save_state_pre_hook(self.save_model_hook)
        self.accelerator.register_load_state_pre_hook(self.load_model_hook)

        self.params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]

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
            num_training_steps=self.max_steps,
        )

        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(self.exp_name)

        self.total_batch_size = (
            self.batch_size
            * self.accelerator.num_processes
            * self.accum_steps
            * len(self.data_conf.valid_layers)
            * self.data_conf.n_layers
        )
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
                dirs = [d for d in os.listdir(self.save_path) if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = os.path.join(self.save_path, dirs[-1]) if dirs else None
            if path is not None:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(path)
                self.global_step = int(path.split("-")[1])

        self.valid_layers = self.train_dataset.valid_layers
        self.n_layers = self.train_dataset.n_layers

        self.wan_params = load_file(self.wan_conf.model_path)
        self.wan_params = {k: v.to(self.dtype) for k, v in self.wan_params.items()}

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(self, models, weights, output_dir):
        state = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            torch.save(state, os.path.join(output_dir, f"model.pth"))
        weights.clear()
    
    def load_model_hook(self, input_dir):
        ckpt_file = os.path.join(
            input_dir,
            f"model.pth"
        )
        state_dict = torch.load(ckpt_file, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)

    def run(self):
        progress_bar = tqdm(
            range(0, self.max_steps),
            initial=self.global_step,
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        all_loss_fn = {name: get_loss_fn(cfg.type) for name, cfg in self.loss_conf.items()}
        train_loss = 0.0
        loss_dict = {}
        gathered_loss_dict = {}

        for _, batch in enumerate(self.train_dataloader):
            if self.global_step >= self.max_steps:
                break

            with self.accelerator.accumulate(self.model):
                batch_dict, input_params = self._process_batch(batch)
                with self.accelerator.autocast():
                    A, B, mask_pred = self.model(
                        input_params,
                        batch_dict["layer_indices"],
                        batch_dict["layer_type"],
                        batch_dict["text_emb"],
                    )

                loss = 0.0
                for key, loss_fn in all_loss_fn.items():
                    loss_cfg = OmegaConf.to_container(self.loss_conf[key], resolve=True)
                    loss_weight = loss_cfg.pop("loss_weight", 1.0)
                    loss_cfg.pop("type", None)
                    if key == "ce_loss":
                        loss_value = loss_fn(mask_pred, batch_dict["mask_gt"], **loss_cfg)
                    elif key == "l2_penalty":
                        loss_value = loss_fn(A, B, **loss_cfg)
                    else:
                        raise NotImplementedError
                    loss += loss_value * loss_weight
                    loss_dict[key] = loss_value.detach()

                for key, loss_value in loss_dict.items():
                    if key not in gathered_loss_dict:
                        gathered_loss_dict["loss/" + key] = 0.0
                    gathered_loss = self.accelerator.gather(loss_value.repeat(self.batch_size)).mean().item() / self.accum_steps
                    gathered_loss_dict["loss/" + key] = gathered_loss
                train_loss += self.accelerator.gather(loss.repeat(self.batch_size)).mean().item() / self.accum_steps

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.params_to_optimize, self.optim_conf.grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                lr = self.lr_scheduler.get_last_lr()[0]
                log_dict = {
                        "train_loss": train_loss, 
                        "lr": lr,
                        "grad_clip": grad_norm.item(),
                    }
                log_dict.update(gathered_loss_dict)
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

    def _process_batch(self, batch):
        eps= 1e-8
        deltaW = batch["lora"].to(self.accelerator.device, dtype=self.dtype, non_blocking=True)
        sampled_layer_indices = batch["layer_depth"].to(self.accelerator.device, non_blocking=True).long()
        sampled_layer_types = batch["layer_type"].to(self.accelerator.device, non_blocking=True).long()
        text_emb = batch["text_emb"].to(self.accelerator.device, dtype=self.dtype, non_blocking=True)

        input_params = []
        for i, (depth, layer_type) in enumerate(zip(
            sampled_layer_indices, sampled_layer_types
        )):
            module_name = self.train_dataset.valid_layers[layer_type.cpu().item()]
            block_name= f"blocks.{depth}.{module_name[:-2]}.{module_name[-1]}.weight"
            input_params.append(self.wan_params[block_name])
        input_params = torch.stack(input_params).to(self.accelerator.device, dtype=self.dtype, non_blocking=True)
        score = deltaW.abs() / (input_params.abs() + eps)

        # if self.accelerator.is_main_process:
        #     i = 0 
        #     for layer_type in sampled_layer_types:
        #         for depth in sampled_layer_indices:
        #             layer_name = self.valid_layers[layer_type.item()]
        #             save_path = os.path.join(
        #                 self.save_path,
        #                 f"{batch['sample_name'][i // 240]}_{depth}_{layer_name}.png"
        #             )
        #             plot_heatmap(
        #                 score[i],
        #                 save_path,
        #                 thresh=self.tau,
        #             )
        #             i += 1
        mask_gt = (score > self.tau).float()


        return dict(layer_indices=sampled_layer_indices, layer_type=sampled_layer_types, mask_gt=mask_gt, text_emb=text_emb), input_params








        