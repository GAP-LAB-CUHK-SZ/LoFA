import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lofa.models.mlp import Mlp, ZeroLinear, softplus_inv
from lofa.models.swiglu_ffn import SwiGLUFFNFused
from lofa.models.block import Block, ConditionalBlock, CrossResidualBlock
from lofa.hyper_modulator_transformer import Learnable2DPos, PatchEmbed


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
        out_act: str = "softplus",
        tau: float = 0.02,
        temperature: float = 0.1,
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
        self.out_act = out_act
        self.gradient_checkpointing = gradient_checkpointing
        self.temperature = temperature
        self.tau = tau

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
                x = checkpoint(lambda _x, _ctx: blk(_x, context=_ctx), x, text, use_reentrant=False)
            else:
                x = blk(x, context=text)

        # for blk in self.blocks:
        #     x = blk(x, context=text)
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

        S = torch.matmul(Bm, A)
        if self.out_act == "sigmoid":
            S = S.pow(2)
            mask_prob = torch.sigmoid((S - self.tau) / self.temperature)
        elif self.out_act == "softplus":
            S = F.softplus(S)
            mask_prob = (S - self.tau) / self.temperature
        else:
            raise NotImplementedError

        return A, Bm, mask_prob

    def forward_features(
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
                x = checkpoint(lambda _x, _ctx: blk(_x, context=_ctx), x, text, use_reentrant=False)
            else:
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

        S = torch.matmul(Bm, A)
        if self.out_act == "sigmoid":
            S = S.pow(2)
            mask_prob = torch.sigmoid((S - self.tau) / self.temperature)
        elif self.out_act == "softplus":
            S = F.softplus(S)
            mask_prob = (S - self.tau) / self.temperature
        else:
            raise NotImplementedError

        return x, S, mask_prob