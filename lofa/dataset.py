import os
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset

"""
we need to handle multi blocks
each block contains self_attn, cross_attn, ffn.0, ffn.2
and each attn contains qkvo

blocks.11.self_attn.k.lora_B.default.weight
blocks.24.ffn.2.lora_B.default.weight
"""
VALID_LAYERS = [
    "self_attn_q", "self_attn_k", "self_attn_v", "self_attn_o",
    "cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_o",
    # "ffn_0", "ffn_2"
]

LAYERS_MAP_A = {
    "self_attn": [32, 1536],
    "cross_attn": [32, 1536],
    "ffn.0": [32, 1536],
    "ffn.2": [32, 8960]
}

class WanHyperDiffDataset(Dataset):
    def __init__(
            self, 
            data_path, 
            meta_path,
            valid_layers=VALID_LAYERS, 
            n_layers=30, 
            rank=32, 
            total_steps=10000000,
            is_train=True,
            aug=None,
            **kwargs
        ):
        self.data_path = data_path
        self.aug_conf = aug
        with open(os.path.join(data_path, meta_path), 'r') as f:
            self.meta_data = json.load(f)
        self.valid_layers = valid_layers
        self.valid_layer_label = {m : i for i, m in enumerate(self.valid_layers)}
        self.n_layers = n_layers
        self.rank = rank
        self._len = len(self.meta_data) if total_steps <= 0 else total_steps
        self.total = len(self.meta_data)
        self.is_train = is_train
        self.use_response_prior = kwargs.get("use_response_prior", False)

    def __len__(self):
        return self._len
    
    @torch.no_grad()
    def __getitem__(self, idx):
        if self.is_train:
            sample_meta = self.meta_data[idx % self.total]
        else:
            sample_meta = self.meta_data[idx]
        sample_name = Path(sample_meta["file_name"]).stem
        lora_path = os.path.join(self.data_path, sample_meta["file_name"])

        with open(lora_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu", weights_only=True)
        
        if self.use_response_prior:
            reponse_prior_path = os.path.join(self.data_path, "stage1", (str(Path(sample_meta['input_emb']).stem) + '.tensors.pth'))
            with open(reponse_prior_path, "rb") as f:
                response_prior = torch.load(f, map_location="cpu", weights_only=True)

        layer_depth = []
        module_type = []
        lora_A = []
        lora_B = []
        input_prior = []
        for layer in self.valid_layers:
            for depth in range(self.n_layers):
                key = f"blocks.{depth}.{layer[:-2]}.{layer[-1]}"
                lora_A.append(state_dict[f"{key}.lora_A.default.weight"] )
                lora_B.append(state_dict[f"{key}.lora_B.default.weight"] )
                layer_depth.append(depth)
                module_type.append(self.valid_layer_label[layer])
                if self.use_response_prior:
                    input_prior.append(response_prior['hidden_states'][f"blocks.{depth}.{layer[:-2]}.{layer[-1]}"])

        lora_A = torch.stack(lora_A)
        lora_B = torch.stack(lora_B)
        total_layers = lora_A.shape[0]
        module_type = torch.Tensor(module_type).long()
        layer_depth = torch.Tensor(layer_depth).long()
        input_prior = torch.stack(input_prior) if self.use_response_prior else None
        
        # get a training sample
        with open(os.path.join(self.data_path, sample_meta["input_emb"]), "rb") as f:
            sample = torch.load(f, map_location="cpu", weights_only=True)
        context = sample["prompt_emb"]
        latent = sample["latents"]

        text_emb = context.unsqueeze(0).repeat(total_layers, 1, 1)
        
        return dict(
            lora_A=lora_A,
            lora_B=lora_B,
            text_emb=text_emb,
            context=context,
            latent=latent,
            input_prior=input_prior,
            sample_name=sample_name,
            layer_type=module_type,
            layer_depth=layer_depth
        )

class WanReconMaskDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        meta_path,
        valid_layers=VALID_LAYERS, 
        n_layers=30,
        total_steps=10000000,
        is_train=True,
        layers_per_sample=-1,
        **kwargs
    ):
        self.data_path = data_path
        with open(os.path.join(data_path, meta_path), 'r') as f:
            self.meta_data = json.load(f)
        self.valid_layers = valid_layers
        self.valid_layer_label = {m : i for i, m in enumerate(self.valid_layers)}
        self.n_layers = n_layers
        self._len = len(self.meta_data) if total_steps <= 0 else total_steps
        self.total = len(self.meta_data)
        self.is_train = is_train
        self.layers_per_sample = layers_per_sample if layers_per_sample > 0 else n_layers * len(valid_layers)
    
    def __len__(self):
        return self._len
    
    @torch.no_grad()
    def __getitem__(self, idx):
        if self.is_train:
            sample_meta = self.meta_data[idx % self.total]
        else:
            sample_meta = self.meta_data[idx]
        sample_name = Path(sample_meta["file_name"]).stem
        lora_path = os.path.join(self.data_path, sample_meta["file_name"])

        with open(lora_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu", weights_only=True)
        layer_depth = []
        module_type = []
        loras = []
        for layer in self.valid_layers:
            for depth in range(self.n_layers):
                # key = f"blocks.{depth}.{layer[:-2]}.{layer[-1]}"
                # W = state_dict[f"{key}.lora_B.default.weight"] @ state_dict[f"{key}.lora_A.default.weight"]
                # loras.append(W)
                layer_depth.append(depth)
                module_type.append(self.valid_layer_label[layer])
        selected_idxs = np.random.choice(len(layer_depth), self.layers_per_sample, replace=False)
        layer_depth = [layer_depth[i] for i in selected_idxs]
        module_type = [module_type[i] for i in selected_idxs]

        for depth, layer_type in zip(layer_depth, module_type):
            layer_name = self.valid_layers[layer_type]
            key = f"blocks.{depth}.{layer_name[:-2]}.{layer_name[-1]}"
            W = state_dict[f"{key}.lora_B.default.weight"] @ state_dict[f"{key}.lora_A.default.weight"]
            loras.append(W)
        loras = torch.stack(loras)
        total_layers = loras.shape[0]
        module_type = torch.Tensor(module_type).long()
        layer_depth = torch.Tensor(layer_depth).long()
        with open(os.path.join(self.data_path, sample_meta["input_emb"]), "rb") as f:
            sample = torch.load(f, map_location="cpu", weights_only=True)

        text_emb = sample["prompt_emb"].unsqueeze(0).repeat(total_layers, 1, 1)
        return dict(
            lora=loras,
            text_emb=text_emb,
            sample_name=sample_name,
            layer_type=module_type,
            layer_depth=layer_depth
        )


def collate_diff(samples):
    batch = {}
    batch["text_emb"] = torch.cat([sample["text_emb"] for sample in samples])
    batch["sample_name"] = [sample["sample_name"] for sample in samples]
    batch["lora_A"] = torch.cat([sample["lora_A"] for sample in samples], dim=0)
    batch["lora_B"] = torch.cat([sample["lora_B"] for sample in samples], dim=0)
    batch["layer_depth"] = torch.cat([sample["layer_depth"] for sample in samples])
    batch["layer_type"] = torch.cat([sample["layer_type"] for sample in samples])
    batch["latent"] = torch.stack([sample["latent"] for sample in samples])
    batch["context"] = torch.stack([sample["context"] for sample in samples])
    batch["input_prior"] = torch.cat([sample["input_prior"] for sample in samples], dim=0) if "input_prior" in samples[0] is not None else None
    return batch


def collate_mask(samples):
    batch = {}
    batch["text_emb"] = torch.cat([sample["text_emb"] for sample in samples])
    batch["sample_name"] = [sample["sample_name"] for sample in samples]
    batch["lora"] = torch.cat([sample["lora"] for sample in samples], dim=0)
    batch["layer_depth"] = torch.cat([sample["layer_depth"] for sample in samples])
    batch["layer_type"] = torch.cat([sample["layer_type"] for sample in samples])
    return batch