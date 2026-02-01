from typing import Callable, Optional, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lofa.models.mlp import Mlp, ZeroLinear, softplus_inv
from lofa.models.swiglu_ffn import SwiGLUFFNFused
from lofa.models.block import Block, ConditionalBlock, CrossResidualBlock

def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)

class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        weight_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 64,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = nn.RMSNorm,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        weight_HW = make_2tuple(weight_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            weight_HW[0] // patch_HW[0],
            weight_HW[1] // patch_HW[1],
        )

        self.weight_size = weight_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert (
            H % patch_H == 0
        ), f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert (
            W % patch_W == 0
        ), f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = (
            Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class Learnable2DPos(nn.Module):
    """Factorized 2D learnable pos embedding: pos = row[r] + col[c]."""
    def __init__(self, gh: int, gw: int, dim: int):
        super().__init__()
        self.row = nn.Parameter(torch.zeros(1, gh, 1, dim))
        self.col = nn.Parameter(torch.zeros(1, 1, gw, dim))
        nn.init.trunc_normal_(self.row, std=0.02)
        nn.init.trunc_normal_(self.col, std=0.02)
        self.gh, self.gw, self.dim = gh, gw, dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], N=gh*gw
        B, N, D = x.shape
        assert D == self.dim and N == self.gh * self.gw
        pos = (self.row + self.col).view(1, N, D)
        return x + pos

class TransformerV2(nn.Module):
    def __init__(self, 
        context_dim: int = 4096,
        in_dim: int = 1536,
        patch_size: int = 64,
        input_size: int = 1536,
        d_model: int = 768,
        n_heads: int = 12,
        n_blocks: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_uniform=False,
        drop_path_rate=0.0,
        lora_rank: int = 32,
        qk_norm: bool = True,
        act_layer=nn.GELU,
        use_cls: bool = True,
        decode_cls: bool = False,
        # layer meta
        num_depth: int = 30,
        num_type: int = 8,
        ffn_layer: str = "mlp",
        alpha_act: str = "softplus",
        gradient_checkpointing: bool = True,
        **kwargs
    ):
        super().__init__()
        self.in_dim = in_dim
        self.context_dim = context_dim
        self.patch_size = patch_size
        self.use_cls = use_cls if not decode_cls else True
        self.num_depth = num_depth
        self.num_type = num_type
        self.decode_cls = decode_cls
        self.alpha_act = alpha_act
        self.gradient_checkpointing = gradient_checkpointing

        self.patch_embed = PatchEmbed(
            weight_size=input_size,
            patch_size=patch_size,
            embed_dim=d_model
        )

        num_patches = self.patch_embed.num_patches
        gh, gw = self.patch_embed.patches_resolution
        self.pos = Learnable2DPos(gh, gw, d_model)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            
        # layer meta embeddings
        self.depth_embed = nn.Embedding(self.num_depth, d_model)
        self.type_embed = nn.Embedding(self.num_type, d_model)
        if self.training:
            nn.init.trunc_normal_(self.depth_embed.weight, std=0.02)
            nn.init.trunc_normal_(self.type_embed.weight, std=0.02)

        # project text to d_model
        self.text_proj = nn.Linear(context_dim, d_model)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * n_blocks
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)
            ]

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError

        # transformer blocks
        blocks_list = [
            ConditionalBlock(
                dim=d_model,
                num_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                qk_norm=qk_norm,
                drop_path=dpr[i]
            )
            for i in range(n_blocks)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = nn.LayerNorm(d_model)

        # head
        self.lora_rank = lora_rank
        
        self.head_A = nn.Linear(d_model, lora_rank * in_dim)
        self.head_B = ZeroLinear(d_model, in_dim * lora_rank)
        self.head_alpha = nn.Linear(d_model, 1)

        if self.training:
            nn.init.zeros_(self.head_alpha.weight)
        alpha_floor = 1e-3
        with torch.no_grad():
            self.head_alpha.bias.fill_(softplus_inv(alpha_floor))

        self.alpha_floor = alpha_floor

    @staticmethod
    def _rms_norm_matrix(W: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # W: [B,H,W]
        return W / (W.pow(2).mean(dim=(1, 2), keepdim=True).add(eps).sqrt())

    def forward(
        self,
        W: torch.Tensor, # (B L) H W
        depth_id: torch.Tensor, # (B L)
        type_id: torch.Tensor,  # (B L)
        text_emb: torch.Tensor, # (B L) N D
    ):
        B = W.shape[0]
        W = W.unsqueeze(1) # (B L) 1 H W
        Wn = self._rms_norm_matrix(W)

        x = self.patch_embed(Wn) # (B L) ph*pw D
        x = self.pos(x)
        # add layer info
        meta = self.depth_embed(depth_id) + self.type_embed(type_id)
        x = x + meta[:, None, :]

        # add cls token
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # [B,1,D]
            x = torch.cat([cls, x], dim=1)

        # text projection
        text = self.text_proj(text_emb)

        for blk in self.blocks:
            if self.gradient_checkpointing:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        for blk in self.blocks:
            x = blk(x, context=text)
        x = self.norm(x)

        # pool
        if self.decode_cls:
            h = x[:, 0]          # [B,D]
        elif self.use_cls:
            h = x.mean(dim=1)
        else:
            h = x[:, 1:].mean(dim=1)    # [B,D]


        A = self.head_A(h).view(B, self.lora_rank, self.in_dim)       # [B, r, in]
        Bm = self.head_B(h).view(B, self.in_dim, self.lora_rank)


        alpha_raw = self.head_alpha(h)              # [B,1]
        if self.alpha_act == "softplus":
            alpha = F.softplus(alpha_raw)
        elif self.alpha_act == "sigmoid":
            alpha = torch.sigmoid(alpha_raw)
        else:
            alpha = alpha_raw

        return A, Bm, alpha
    

class TransformerV2_stage2(nn.Module):
    def __init__(self, 
        context_dim: int = 4096,
        in_dim: int = 1536,
        patch_size: int = 64,
        input_size: int = 1536,
        d_model: int = 768,
        n_heads: int = 12,
        n_blocks: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_uniform=False,
        drop_path_rate=0.0,
        lora_rank: int = 32,
        qk_norm: bool = True,
        act_layer=nn.GELU,
        use_cls: bool = True,
        decode_cls: bool = False,
        # layer meta
        num_depth: int = 30,
        num_type: int = 8,
        ffn_layer: str = "mlp",
        out_act: str = "softplus",
        alpha_act: str = "softplus",
        gradient_checkpointing: bool = True,
        # stage1 feature injection
        cross_attn_idxs: Optional[list[int]] = [3, 7],
        **kwargs
    ):
        super().__init__()
        self.in_dim = in_dim
        self.context_dim = context_dim
        self.patch_size = patch_size
        self.use_cls = use_cls if not decode_cls else True
        self.num_depth = num_depth
        self.num_type = num_type
        self.decode_cls = decode_cls
        self.out_act = out_act
        self.gradient_checkpointing = gradient_checkpointing
        self.alpha_act = alpha_act

        self.patch_embed = PatchEmbed(
            weight_size=input_size,
            patch_size=patch_size,
            embed_dim=d_model
        )

        num_patches = self.patch_embed.num_patches
        gh, gw = self.patch_embed.patches_resolution
        self.pos = Learnable2DPos(gh, gw, d_model)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            
        # layer meta embeddings
        self.depth_embed = nn.Embedding(self.num_depth, d_model)
        self.type_embed = nn.Embedding(self.num_type, d_model)
        if self.training:
            nn.init.trunc_normal_(self.depth_embed.weight, std=0.02)
            nn.init.trunc_normal_(self.type_embed.weight, std=0.02)

        # project text to d_model
        self.text_proj = nn.Linear(context_dim, d_model)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * n_blocks
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)
            ]

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError

        # transformer blocks
        blocks_list = [
            ConditionalBlock(
                dim=d_model,
                num_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                qk_norm=qk_norm,
                drop_path=dpr[i]
            )
            for i in range(n_blocks)
        ]
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = nn.LayerNorm(d_model)

        # stage 1 feature cross attention blocks
        self.cross_attn_blocks = nn.ModuleDict()
        for idx in cross_attn_idxs:
            self.cross_attn_blocks[str(idx)] = CrossResidualBlock(
                dim=d_model,
                num_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                ffn_bias=ffn_bias,
                proj_bias=proj_bias,
                norm_layer=nn.LayerNorm,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                qk_norm=qk_norm,
                drop_path=dpr[idx]
            )

        self.cross_attn_idxs = cross_attn_idxs
        # head
        self.lora_rank = lora_rank
        
        self.head_A = nn.Linear(d_model, lora_rank * in_dim)
        self.head_B = ZeroLinear(d_model, in_dim * lora_rank)
        self.head_alpha = nn.Linear(d_model, 1)

        if self.training:
            nn.init.zeros_(self.head_alpha.weight)
        alpha_floor = 1e-3
        with torch.no_grad():
            self.head_alpha.bias.fill_(softplus_inv(alpha_floor))

        self.alpha_floor = alpha_floor
        

    @staticmethod
    def _rms_norm_matrix(W: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # W: [B,H,W]
        return W / (W.pow(2).mean(dim=(1, 2), keepdim=True).add(eps).sqrt())

    def forward(
        self,
        W: torch.Tensor, # (B L) H W
        depth_id: torch.Tensor, # (B L)
        type_id: torch.Tensor,  # (B L)
        text_emb: torch.Tensor, # (B L) N D
        input_prior: Optional[torch.Tensor] = None, # (B L) N D
    ):
        B = W.shape[0]
        W = W.unsqueeze(1) # (B L) 1 H W
        Wn = self._rms_norm_matrix(W)

        x = self.patch_embed(Wn) # (B L) ph*pw D
        x = self.pos(x)
        # add layer info
        meta = self.depth_embed(depth_id) + self.type_embed(type_id)
        x = x + meta[:, None, :]

        # add cls token
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # [B,1,D]
            x = torch.cat([cls, x], dim=1)

        # text projection
        text = self.text_proj(text_emb)

        for i, blk in enumerate(self.blocks):
            if self.gradient_checkpointing:
                x = checkpoint(lambda _x, _ctx: blk(_x, context=_ctx), x, text, use_reentrant=False)
            else:
                x = blk(x, context=text)

            if i in self.cross_attn_idxs:
                cross_blk = self.cross_attn_blocks[str(i)]
                if self.gradient_checkpointing:
                    x = checkpoint(lambda _x, _ctx: cross_blk(_x, context=_ctx), x, input_prior, use_reentrant=False)
                else:
                    x = cross_blk(x, context=input_prior)
        x = self.norm(x)

        # pool
        if self.decode_cls:
            h = x[:, 0]          # [B,D]
        elif self.use_cls:
            h = x.mean(dim=1)
        else:
            h = x[:, 1:].mean(dim=1)    # [B,D]


        A = self.head_A(h).view(B, self.lora_rank, self.in_dim)       # [B, r, in]
        Bm = self.head_B(h).view(B, self.in_dim, self.lora_rank)

        alpha_raw = self.head_alpha(h)              # [B,1]
        if self.alpha_act == "softplus":
            alpha = F.softplus(alpha_raw)
        elif self.alpha_act == "sigmoid":
            alpha = torch.sigmoid(alpha_raw)
        else:
            alpha = alpha_raw

        return A, Bm, alpha