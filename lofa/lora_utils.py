import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from operator import attrgetter
from functools import partial
from typing import Any

import torch
from peft import PeftModel
import matplotlib.pyplot as plt

def plot_heatmap(x: torch.Tensor, save_path, thresh=0.02):
    x_np = x.float().cpu().numpy()
    color = np.empty(x_np.shape + (3,), dtype=float)
    mask_white = np.abs(x_np) < thresh
    color[mask_white] = [1, 1, 1]

    mask_red = np.abs(x_np) >= thresh
    color[mask_red] = [1, 0, 0]

    plt.imshow(color, interpolation='nearest')
    plt.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def add_lora_hooks(model, module_names, layer_indices, A: torch.Tensor, B: torch.Tensor, scaling, input_dropout, training):
    def lora_hook(module, args, output):
        if isinstance(output, tuple):
            model_out = output[0]
        else:
            model_out = output

        x = args[0].to(A.dtype)
        bs, inp_len = x.shape[:2]

        # A and B repeat for each input token
        lora_A = A.repeat_interleave(inp_len, dim=0)
        lora_B = B.repeat_interleave(inp_len, dim=0)
        x = x.reshape(bs * inp_len, 1, -1)
        delta_x = torch.bmm(torch.bmm(F.dropout(x, input_dropout, training), lora_A), lora_B) * scaling

        newoutput = model_out + delta_x.reshape(bs, inp_len, -1).to(model_out.dtype)
        if isinstance(output, tuple):
            return (newoutput, *output[1:])
        else:
            return newoutput
    pass

def remove_hooks_(module):
    module._forward_hooks = OrderedDict()
    module._foward_pre_hooks = OrderedDict()

def apply_custom_hooks_at_layers_(
    model,
    module_names,
    layer_indices,
    pre_hook=None,
    post_hook=None,
    remove_other_hooks=False,
):
    if "block" in module_names:
        assert len(module_names) == 1

    layers =model
    out_handles = []
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for mname in module_names:
            handles = apply_hook_to_layer_(
                layer,
                mname,
                pre_hook=pre_hook,
                post_hook=post_hook,
                remove_other_hooks=remove_other_hooks,
            )
            out_handles += handles
    return out_handles

def apply_hook_to_layer_(
    layer,
    mname,
    pre_hook=None,
    post_hook=None,
    remove_other_hooks=False,
):
    # assert mname in ["block", "mlp", "self_attn"]
    assert (pre_hook is not None) or (post_hook is not None)
    pre_hook_handle = None
    post_hook_handle = None
    module = attrgetter(mname)(layer)
    if remove_other_hooks:
        remove_hooks_(module)
    if pre_hook is not None:
        pre_hook_handle = module.register_forward_pre_hook(pre_hook)
    if post_hook is not None:
        post_hook_handle = module.register_forward_hook(post_hook)
    return pre_hook_handle, post_hook_handle