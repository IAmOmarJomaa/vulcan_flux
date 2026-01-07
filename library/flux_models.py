import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class FluxParams:
    depth: int
    depth_single_blocks: int
    num_heads: int
    hidden_size: int
    in_channels: int
    vec_in_dim: int
    context_dim: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    guidance_embed: bool = True

# ... (EmbedND, MLPEmbedder, RMSNorm, QKNorm classes are standard, keep them or use the versions below) ...

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = len(self.axes_dim)
        emb = torch.cat([nn.functional.embedding(ids[..., i], self.create_embedding(i)) for i in range(n_axes)], dim=-1)
        return emb.unsqueeze(1)
    def create_embedding(self, idx):
        return torch.randn(self.axes_dim[idx], self.dim // len(self.axes_dim))

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_norm(q), self.key_norm(k)

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        # VULCAN FIX: Naming layers 'lin' to match checkpoint keys
        self.img_mod = nn.Module()
        self.img_mod.lin = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.txt_mod = nn.Module()
        self.txt_mod.lin = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
        self.txt_attn_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio), bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size, bias=True)
        )
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio), bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size, bias=True)
        )
    def forward(self, img, txt, vec, pe):
        return img, txt

class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, bias=qkv_bias)
        # VULCAN FIX: 15360 dimension match
        self.linear2 = nn.Linear(self.mlp_hidden_dim + hidden_size, hidden_size, bias=qkv_bias)
        self.norm = QKNorm(hidden_size // num_heads)
        self.modulation = nn.Module()
        self.modulation.lin = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
    def forward(self, x, vec, pe):
        return x

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
    def forward(self, x, vec):
        return self.linear(self.norm_final(x))

class Flux(nn.Module):
    def __init__(self, params: FluxParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=params.hidden_size) if params.guidance_embed else nn.Identity()
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=params.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, params.hidden_size)
        self.img_in = nn.Linear(self.in_channels, params.hidden_size, bias=True)
        self.txt_in = nn.Linear(params.context_dim, params.hidden_size, bias=True)
        self.double_blocks = nn.ModuleList([DoubleStreamBlock(params.hidden_size, params.num_heads, params.mlp_ratio, params.qkv_bias) for _ in range(params.depth)])
        self.single_blocks = nn.ModuleList([SingleStreamBlock(params.hidden_size, params.num_heads, params.mlp_ratio, params.qkv_bias) for _ in range(params.depth_single_blocks)])
        self.final_layer = LastLayer(params.hidden_size, 1, self.out_channels)
    def forward(self, x):
        pass

# VULCAN FIX: Config keys must match what flux_utils.py asks for ("dev", "schnell")
configs = {
    "dev": FluxParams(depth=19, depth_single_blocks=38, num_heads=24, hidden_size=3072, in_channels=64, vec_in_dim=768, context_dim=4096, mlp_ratio=4.0, qkv_bias=True, guidance_embed=True),
    "schnell": FluxParams(depth=19, depth_single_blocks=38, num_heads=24, hidden_size=3072, in_channels=64, vec_in_dim=768, context_dim=4096, mlp_ratio=4.0, qkv_bias=True, guidance_embed=True),
}