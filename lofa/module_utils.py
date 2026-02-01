
import torch.nn as nn
memory_efficient_attention = None
try:
    import xformers
except:
    pass

try:
    from xformers.ops import memory_efficient_attention
    XFORMERS_AVAIL = True
except:
    memory_efficient_attention = None
    XFORMERS_AVAIL = False


def default(val, d):
    return val if val is not None else d

def get_ac_fn(fname):
    if fname == "Tanh":
        return nn.Tanh
    elif fname == "SiLU":
        return nn.SiLU
    elif fname == "sigmoid":
        return nn.Sigmoid
    elif fname == "GeLU":
        return nn.GELU
    else:
        raise NotImplementedError