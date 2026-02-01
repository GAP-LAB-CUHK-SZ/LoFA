import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from lofa.module_utils import (
    default,
    XFORMERS_AVAIL,
    memory_efficient_attention
)

XFORMERS_AVAILABLE = False

XFORMERS_ENABLED = False
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=(
                    (attn_mask)[:, None].repeat(1, self.num_heads, 1, 1)
                    if attn_mask is not None
                    else None
                ),
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
        
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        # separate projections for cross-attn
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        attn_mask: Tensor = None,
    ) -> Tensor:
        B, Nq, C = x.shape
        assert context.dim() == 3 and context.shape[0] == B and context.shape[2] == C
        Nk = context.shape[1]

        # Q: [B,H,Nq,Dh]
        q = (
            self.q(x)
            .reshape(B, Nq, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        # K,V: [B,H,Nk,Dh]
        kv = (
            self.kv(context)
            .reshape(B, Nk, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        q, k = self.q_norm(q), self.k_norm(k)

        if attn_mask is not None:
            attn_mask_ = self._expand_attn_mask(attn_mask)
        else:
            attn_mask_ = None

        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask_,
            )  # [B,H,Nq,Dh]
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B,H,Nq,Nk]
            if attn_mask_ is not None:
                # Prefer additive mask (-inf) for correctness; if bool, convert outside.
                attn = attn + attn_mask_.to(attn.dtype)
            attn = attn.softmax(dim=-1)
            out = attn @ v  # [B,H,Nq,Dh]

        out = out.transpose(1, 2).reshape(B, Nq, C)
        out = self.proj(out)

        return out


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


ATTN_FUNCTION = {
    'torch': F.scaled_dot_product_attention,
    'xformers': memory_efficient_attention
}

def low_version_attention(query, key, value, attn_bias=None):
    scale = 1 / query.shape[-1] ** 0.5
    query = query * scale
    attn = torch.matmul(query, key.transpose(-2, -1))
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    return attn @ value

class BaseAttention(nn.Module):
    def __init__(self, q_dim, num_heads, head_dim, kv_dim=None, bias_q=False, bias_kv=False, bias_out=False):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = torch.nn.Linear(q_dim, dim_inner, bias=bias_q)
        self.to_k = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_v = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_out = torch.nn.Linear(dim_inner, q_dim, bias=bias_out)

    def torch_forward(self, hidden_states, encoder_hidden_states=None, attn_mask=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size = encoder_hidden_states.shape[0]

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)


        hidden_states = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        hidden_states = hidden_states.to(q.dtype)

        hidden_states = self.to_out(hidden_states)

        return hidden_states
    
    def xformers_forward(self, hidden_states, encoder_hidden_states=None, attn_mask=None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        q = rearrange(q, "b f (n d) -> (b n) f d", n=self.num_heads)
        k = rearrange(k, "b f (n d) -> (b n) f d", n=self.num_heads)
        v = rearrange(v, "b f (n d) -> (b n) f d", n=self.num_heads)

        if attn_mask is not None:
            hidden_states = low_version_attention(q, k, v, attn_bias=attn_mask)
        else:
            import xformers.ops as xops
            hidden_states = xops.memory_efficient_attention(q, k, v)
        hidden_states = rearrange(hidden_states, "(b n) f d -> b f (n d)", n=self.num_heads)

        hidden_states = hidden_states.to(q.dtype)
        hidden_states = self.to_out(hidden_states)

        return hidden_states
    
    def forward(self, hidden_states, encoder_hidden_states=None, attn_mask=None):
        return self.torch_forward(hidden_states, encoder_hidden_states=encoder_hidden_states, attn_mask=attn_mask)