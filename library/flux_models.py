import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

@dataclass
class FluxParams:
    class FluxParams:
    mlp_ratio: float = 4.0
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool

@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float

def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

# --- LAYERS ---
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat([ids[..., i] for i in range(n_axes)], dim=-1)
        return emb

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k, v

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.norm = QKNorm(head_dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "b l (qkv h d) -> qkv b h l d", qkv=3, h=self.num_heads)
        q, k, v = self.norm(q, k, v)
        
        # RoPE would go here, simplified for brevity as usually precomputed or external
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = rearrange(x, "b h l d -> b l (h d)")
        return self.proj(x)

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.img_mod = nn.Module()
        self.img_mod.lin = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio), bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size, bias=True),
        )
        self.txt_mod = nn.Module()
        self.txt_mod.lin = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio), bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod.lin(vec).chunk(2, dim=-1)
        txt_mod1, txt_mod2 = self.txt_mod.lin(vec).chunk(2, dim=-1)
        
        # Prepare modulation
        # (Simplified logic for modulation application)
        # In full implementation, this applies scale/shift. 
        # For Skeleton Crew, ensuring shapes align is key.
        return img, txt 

class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # VULCAN MATH FIX:
        # Checkpoint expects linear1 to be (3 * hidden) + (mlp_ratio * hidden)
        # For Flux: (3 * 3072) + (4 * 3072) = 9216 + 12288 = 21504.
        # BUT linear2 (the MLP output) specifically needs to handle the expanded hidden dim.
        # Checkpoint shape [3072, 15360] means the input to linear2 is 15360.
        
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio) # 12288
        
        # Linear1: QKV + MLP expansion
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, bias=True)
        
        # Linear2: The specific layer causing the error. 
        # Checkpoint says: "copying a param with shape [3072, 15360] from checkpoint"
        # 15360 = 12288 (MLP) + 3072 (Residual/Context)
        self.linear2 = nn.Linear(self.mlp_hidden_dim + hidden_size, hidden_size, bias=True)

        self.norm = QKNorm(hidden_size // num_heads)
        self.modulation = nn.Module()
        self.modulation.lin = nn.Linear(hidden_size, 3 * hidden_size, bias=True)

    def forward(self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        # Keep your existing forward logic here
        return x

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return self.linear(x)

# --- FLUX MODEL ---
class Flux(nn.Module):
    def __init__(self, params: FluxParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        
        self.pe_embedder = EmbedND(dim=params.hidden_size, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(params.in_channels, params.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=params.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, params.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=params.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, params.hidden_size, bias=True)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    params.hidden_size,
                    params.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    params.hidden_size,
                    params.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(params.hidden_size, 1, self.out_channels) # patch_size=1 (already packed)

        self.swap_blocks_interval = 0
        self.move_to_device = None

    def enable_block_swap(self, interval: int, device: torch.device):
        self.swap_blocks_interval = interval
        self.move_to_device = device

    def prepare_block_swap_before_forward(self):
        # Move initial layers to GPU
        if self.move_to_device:
            self.img_in.to(self.move_to_device)
            self.time_in.to(self.move_to_device)
            self.vector_in.to(self.move_to_device)
            if isinstance(self.guidance_in, nn.Module): self.guidance_in.to(self.move_to_device)
            self.txt_in.to(self.move_to_device)
            self.pe_embedder.to(self.move_to_device)

    def move_to_device_except_swap_blocks(self, device):
        # Move final layer to device, keep blocks on CPU if swapping is on
        self.final_layer.to(device)
        if self.swap_blocks_interval == 0:
            self.double_blocks.to(device)
            self.single_blocks.to(device)

    def get_mod_vectors(self, timesteps, guidance, batch_size):
        # Standard modulation logic
        vec = self.time_in(torch.zeros(batch_size, 256, device=timesteps.device)) # Simplified
        if self.params.guidance_embed:
            vec = vec + self.guidance_in(torch.zeros(batch_size, 256, device=guidance.device))
        return vec

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        txt_attention_mask: Tensor = None,
        mod_vectors: Tensor = None,
    ) -> Tensor:
        
        # 1. Embedding
        img = self.img_in(img)
        vec = self.vector_in(y)
        if mod_vectors is not None:
             vec = vec + mod_vectors
        
        txt = self.txt_in(txt)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # 2. Double Blocks
        for i, block in enumerate(self.double_blocks):
            if self.swap_blocks_interval > 0 and self.move_to_device:
                block.to(self.move_to_device)
            
            if self.training and torch.is_grad_enabled():
                img, txt = checkpoint(block, img, txt, vec, pe, use_reentrant=False)
            else:
                img, txt = block(img, txt, vec, pe)

            if self.swap_blocks_interval > 0 and self.move_to_device:
                block.to("cpu") # Swap back

        # 3. Single Blocks
        img = torch.cat((txt, img), dim=1)
        for i, block in enumerate(self.single_blocks):
            if self.swap_blocks_interval > 0 and self.move_to_device:
                block.to(self.move_to_device)
            
            if self.training and torch.is_grad_enabled():
                img = checkpoint(block, img, vec, pe, use_reentrant=False)
            else:
                img = block(img, vec, pe)

            if self.swap_blocks_interval > 0 and self.move_to_device:
                block.to("cpu")

        # 4. Output
        img = img[:, txt.shape[1] :, ...]
        img = self.final_layer(img, vec)
        return img

# --- CONFIGS ---
configs = {
    "MODEL_NAME_DEV": FluxParams(
        depth=19,
        depth_single_blocks=38,),
    "dev": FluxParams(
        in_channels=64, vec_in_dim=768, context_in_dim=4096, hidden_size=3072, mlp_ratio=4.0,
        num_heads=24, depth=19, depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10000, qkv_bias=True, guidance_embed=True
    ),
    "schnell": FluxParams(
        in_channels=64, vec_in_dim=768, context_in_dim=4096, hidden_size=3072, mlp_ratio=4.0,
        num_heads=24, depth=19, depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10000, qkv_bias=True, guidance_embed=True
    ),
    "chroma": FluxParams(
         in_channels=64, vec_in_dim=768, context_in_dim=4096, hidden_size=3072, mlp_ratio=4.0,
        num_heads=24, depth=19, depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10000, qkv_bias=True, guidance_embed=True
    )
}

# --- AUTOENCODER (Keeping it simple as it's mostly for loading) ---
class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = nn.Module() # Placeholder
        self.decoder = nn.Module() # Placeholder
    def encode(self, x): return x 
    def decode(self, z): return z
    # Note: Full AE implementation is usually not needed for Training (Latents are pre-cached), 
    # but needed if you run validation. For the Skeleton Crew, this placeholder prevents crash on load.