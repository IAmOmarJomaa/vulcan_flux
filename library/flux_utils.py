from huggingface_hub import hf_hub_download
import torch
import os
from safetensors.torch import load_file
from library import flux_models
from library.utils import setup_logging
import logging
from typing import Tuple, List, Optional, Union
from safetensors import safe_open
from dataclasses import replace
from transformers import CLIPConfig, CLIPTextModel, T5Config, T5EncoderModel
from accelerate import init_empty_weights
from library.safetensors_utils import load_safetensors
import einops

setup_logging()
logger = logging.getLogger(__name__)

MODEL_VERSION_FLUX_V1 = "flux1"
MODEL_NAME_DEV = "dev"
MODEL_NAME_SCHNELL = "schnell"
MODEL_VERSION_CHROMA = "chroma"

def analyze_checkpoint_state(ckpt_path: str) -> Tuple[bool, bool, Tuple[int, int], List[str]]:
    # VULCAN FIX: Auto-download from HF if not local
    if not os.path.exists(ckpt_path) and not os.path.exists(os.path.join(ckpt_path, "transformer")):
        logger.info(f"'{ckpt_path}' not found locally. Attempting HF download...")
        try:
            # Target the standard BFL file
            ckpt_path = hf_hub_download(repo_id=ckpt_path, filename="flux1-dev.safetensors")
            logger.info(f"Downloaded to: {ckpt_path}")
        except Exception as e:
            logger.warning(f"Download attempt failed: {e}. Assuming path is valid and hoping for the best...")

    # Existing code continues below...
    ckpt_path = ckpt_path.strip('"').strip("'")
    # ...
    ckpt_path = ckpt_path.strip('"').strip("'")
    
    # VULCAN FIX: Handle directory paths for Kaggle
    if os.path.isdir(ckpt_path):
        potential = os.path.join(ckpt_path, "transformer", "diffusion_pytorch_model-00001-of-00003.safetensors")
        if os.path.exists(potential):
            ckpt_path = potential
            
    logger.info(f"Checking the state dict: Diffusers or BFL, dev or schnell")
    
    # Handle multi-part checkpoints
    if "00001-of-00003" in ckpt_path:
        ckpt_paths = [ckpt_path.replace("00001-of-00003", f"0000{i}-of-00003") for i in range(1, 4)]
    else:
        ckpt_paths = [ckpt_path]

    keys = []
    for p in ckpt_paths:
        with safe_open(p, framework="pt") as f:
            keys.extend(f.keys())

    # VULCAN FIX: Robust prefix stripping for analysis
    if keys[0].startswith("model.diffusion_model."):
        keys = [key.replace("model.diffusion_model.", "") for key in keys]

    is_diffusers = "transformer_blocks.0.attn.add_k_proj.bias" in keys
    is_schnell = not ("guidance_in.in_layer.bias" in keys or "time_text_embed.guidance_embedder.linear_1.bias" in keys)
    
    # Check number of blocks
    if not is_diffusers:
        max_double = max([int(k.split(".")[1]) for k in keys if k.startswith("double_blocks.") and k.endswith(".img_attn.proj.bias")])
        max_single = max([int(k.split(".")[1]) for k in keys if k.startswith("single_blocks.") and k.endswith(".modulation.lin.bias")])
    else:
        max_double = max([int(k.split(".")[1]) for k in keys if k.startswith("transformer_blocks.") and k.endswith(".attn.add_k_proj.bias")])
        max_single = max([int(k.split(".")[1]) for k in keys if k.startswith("single_transformer_blocks.") and k.endswith(".attn.to_k.bias")])

    num_double = max_double + 1
    num_single = max_single + 1
    
    return is_diffusers, is_schnell, (num_double, num_single), ckpt_paths

def load_flow_model(ckpt_path: str, dtype, device, disable_mmap=False, model_type="flux"):
    # VULCAN FIX: Ensure path is clean
    ckpt_path = ckpt_path.strip('"').strip("'")
    
    if model_type == "flux":
        is_diffusers, is_schnell, (num_double_blocks, num_single_blocks), ckpt_paths = analyze_checkpoint_state(ckpt_path)
        name = MODEL_NAME_DEV if not is_schnell else MODEL_NAME_SCHNELL
        
        logger.info(f"Building Flux model {name} from {'Diffusers' if is_diffusers else 'BFL'} checkpoint")
        with torch.device("meta"):
            params = flux_models.configs[name].params
            if params.depth != num_double_blocks:
                params = replace(params, depth=num_double_blocks)
            if params.depth_single_blocks != num_single_blocks:
                params = replace(params, depth_single_blocks=num_single_blocks)
            
            model = flux_models.Flux(params)
            if dtype is not None:
                model = model.to(dtype)

        logger.info(f"Loading state dict from {ckpt_path}")
        sd = {}
        for p in ckpt_paths:
            sd.update(load_safetensors(p, device=device, disable_mmap=disable_mmap, dtype=dtype))

        if is_diffusers:
            logger.info("Converting Diffusers to BFL")
            sd = convert_diffusers_sd_to_bfl(sd, num_double_blocks, num_single_blocks)

        # VULCAN CRITICAL FIX: The "Ghost Model" Prefix Stripper (Flux)
        keys_to_rename = [k for k in sd.keys() if k.startswith("model.diffusion_model.")]
        for key in keys_to_rename:
            new_key = key.replace("model.diffusion_model.", "")
            sd[new_key] = sd.pop(key)
        if keys_to_rename:
            logger.info(f"Sanitized {len(keys_to_rename)} keys (Flux mode).")

        info = model.load_state_dict(sd, strict=False, assign=True)
        logger.info(f"Loaded Flux: {info}")
        return is_schnell, model

    elif model_type == "chroma":
        from . import chroma_models
        logger.info("Building Chroma model")
        with torch.device("meta"):
            model = chroma_models.Chroma(chroma_models.chroma_params)
            if dtype is not None:
                model = model.to(dtype)

        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

        # VULCAN CRITICAL FIX: The "Ghost Model" Prefix Stripper (Chroma)
        keys_to_rename = [k for k in sd.keys() if k.startswith("model.diffusion_model.")]
        for key in keys_to_rename:
            new_key = key.replace("model.diffusion_model.", "")
            sd[new_key] = sd.pop(key)
        if keys_to_rename:
            logger.info(f"Sanitized {len(keys_to_rename)} keys (Chroma mode).")

        info = model.load_state_dict(sd, strict=False, assign=True)
        logger.info(f"Loaded Chroma: {info}")
        return False, model

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def load_ae(ckpt_path, dtype, device, disable_mmap=False):
    logger.info("Building AutoEncoder")
    with torch.device("meta"):
        ae = flux_models.AutoEncoder(flux_models.configs[MODEL_NAME_DEV].ae_params).to(dtype)
    
    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = ae.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded AE: {info}")
    return ae

# --- Helper Classes for CLIP/T5 (Standard) ---
def dummy_clip_l() -> torch.nn.Module:
    return DummyCLIPL()

class DummyTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Parameter(torch.zeros(1))

class DummyCLIPL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_shape = (77, 1)
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        self.text_model = DummyTextModel()
    @property
    def device(self): return self.dummy_param.device
    @property
    def dtype(self): return self.dummy_param.dtype
    def forward(self, *args, **kwargs):
        batch_size = args[0].shape[0] if args else 1
        return {"pooler_output": torch.zeros(batch_size, *self.output_shape, device=self.device, dtype=self.dtype)}

def load_clip_l(ckpt_path, dtype, device, disable_mmap=False, state_dict=None):
    logger.info("Building CLIP-L")
    # Reduced config for brevity, sufficient for loading weights
    CLIPL_CONFIG = {
        "architectures": ["CLIPModel"], "model_type": "clip_text_model", "hidden_size": 768,
        "intermediate_size": 3072, "num_attention_heads": 12, "num_hidden_layers": 12, "vocab_size": 49408
    }
    config = CLIPConfig(**CLIPL_CONFIG)
    with init_empty_weights():
        clip = CLIPTextModel._from_config(config)

    if state_dict is None:
        sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    else:
        sd = state_dict
    info = clip.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded CLIP-L: {info}")
    return clip

def load_t5xxl(ckpt_path, dtype, device, disable_mmap=False, state_dict=None):
    # Simplified config for T5XXL
    T5_CONFIG_JSON = """{"architectures":["T5EncoderModel"],"d_model":4096,"num_layers":24,"num_heads":64,"d_kv":64,"d_ff":10240,"vocab_size":32128,"model_type":"t5","is_encoder_decoder":true,"is_gated_act":true}"""
    import json
    config = T5Config(**json.loads(T5_CONFIG_JSON))
    with init_empty_weights():
        t5xxl = T5EncoderModel._from_config(config)
    
    if state_dict is None:
        sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    else:
        sd = state_dict
    info = t5xxl.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded T5xxl: {info}")
    return t5xxl

def get_t5xxl_actual_dtype(t5xxl):
    return t5xxl.encoder.block[0].layer[0].SelfAttention.q.weight.dtype

def prepare_img_ids(batch_size, h, w):
    img_ids = torch.zeros(h, w, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w)[None, :]
    return einops.repeat(img_ids, "h w c -> b (h w) c", b=batch_size)

def unpack_latents(x, h, w):
    return einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=2, pw=2)

def pack_latents(x):
    return einops.rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

# --- BFL to Diffusers Conversion Helpers (Must include for compatibility) ---
NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38
BFL_TO_DIFFUSERS_MAP = {
    "time_in.in_layer.weight": ["time_text_embed.timestep_embedder.linear_1.weight"],
    "time_in.in_layer.bias": ["time_text_embed.timestep_embedder.linear_1.bias"],
    "time_in.out_layer.weight": ["time_text_embed.timestep_embedder.linear_2.weight"],
    "time_in.out_layer.bias": ["time_text_embed.timestep_embedder.linear_2.bias"],
    "vector_in.in_layer.weight": ["time_text_embed.text_embedder.linear_1.weight"],
    "vector_in.in_layer.bias": ["time_text_embed.text_embedder.linear_1.bias"],
    "vector_in.out_layer.weight": ["time_text_embed.text_embedder.linear_2.weight"],
    "vector_in.out_layer.bias": ["time_text_embed.text_embedder.linear_2.bias"],
    "guidance_in.in_layer.weight": ["time_text_embed.guidance_embedder.linear_1.weight"],
    "guidance_in.in_layer.bias": ["time_text_embed.guidance_embedder.linear_1.bias"],
    "guidance_in.out_layer.weight": ["time_text_embed.guidance_embedder.linear_2.weight"],
    "guidance_in.out_layer.bias": ["time_text_embed.guidance_embedder.linear_2.bias"],
    "txt_in.weight": ["context_embedder.weight"],
    "txt_in.bias": ["context_embedder.bias"],
    "img_in.weight": ["x_embedder.weight"],
    "img_in.bias": ["x_embedder.bias"],
    "double_blocks.().img_mod.lin.weight": ["norm1.linear.weight"],
    "double_blocks.().img_mod.lin.bias": ["norm1.linear.bias"],
    "double_blocks.().txt_mod.lin.weight": ["norm1_context.linear.weight"],
    "double_blocks.().txt_mod.lin.bias": ["norm1_context.linear.bias"],
    "double_blocks.().img_attn.qkv.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
    "double_blocks.().img_attn.qkv.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"],
    "double_blocks.().txt_attn.qkv.weight": ["attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"],
    "double_blocks.().txt_attn.qkv.bias": ["attn.add_q_proj.bias", "attn.add_k_proj.bias", "attn.add_v_proj.bias"],
    "double_blocks.().img_attn.norm.query_norm.scale": ["attn.norm_q.weight"],
    "double_blocks.().img_attn.norm.key_norm.scale": ["attn.norm_k.weight"],
    "double_blocks.().txt_attn.norm.query_norm.scale": ["attn.norm_added_q.weight"],
    "double_blocks.().txt_attn.norm.key_norm.scale": ["attn.norm_added_k.weight"],
    "double_blocks.().img_mlp.0.weight": ["ff.net.0.proj.weight"],
    "double_blocks.().img_mlp.0.bias": ["ff.net.0.proj.bias"],
    "double_blocks.().img_mlp.2.weight": ["ff.net.2.weight"],
    "double_blocks.().img_mlp.2.bias": ["ff.net.2.bias"],
    "double_blocks.().txt_mlp.0.weight": ["ff_context.net.0.proj.weight"],
    "double_blocks.().txt_mlp.0.bias": ["ff_context.net.0.proj.bias"],
    "double_blocks.().txt_mlp.2.weight": ["ff_context.net.2.weight"],
    "double_blocks.().txt_mlp.2.bias": ["ff_context.net.2.bias"],
    "double_blocks.().img_attn.proj.weight": ["attn.to_out.0.weight"],
    "double_blocks.().img_attn.proj.bias": ["attn.to_out.0.bias"],
    "double_blocks.().txt_attn.proj.weight": ["attn.to_add_out.weight"],
    "double_blocks.().txt_attn.proj.bias": ["attn.to_add_out.bias"],
    "single_blocks.().modulation.lin.weight": ["norm.linear.weight"],
    "single_blocks.().modulation.lin.bias": ["norm.linear.bias"],
    "single_blocks.().linear1.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "proj_mlp.weight"],
    "single_blocks.().linear1.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias", "proj_mlp.bias"],
    "single_blocks.().linear2.weight": ["proj_out.weight"],
    "single_blocks.().norm.query_norm.scale": ["attn.norm_q.weight"],
    "single_blocks.().norm.key_norm.scale": ["attn.norm_k.weight"],
    "single_blocks.().linear2.bias": ["proj_out.bias"],
    "final_layer.linear.weight": ["proj_out.weight"],
    "final_layer.linear.bias": ["proj_out.bias"],
    "final_layer.adaLN_modulation.1.weight": ["norm_out.linear.weight"],
    "final_layer.adaLN_modulation.1.bias": ["norm_out.linear.bias"],
}

def make_diffusers_to_bfl_map(num_double, num_single):
    diffusers_to_bfl = {}
    for b in range(num_double):
        for key, weights in BFL_TO_DIFFUSERS_MAP.items():
            if key.startswith("double_blocks."):
                prefix = f"transformer_blocks.{b}."
                for i, w in enumerate(weights):
                    diffusers_to_bfl[f"{prefix}{w}"] = (i, key.replace("()", f"{b}"))
    for b in range(num_single):
        for key, weights in BFL_TO_DIFFUSERS_MAP.items():
            if key.startswith("single_blocks."):
                prefix = f"single_transformer_blocks.{b}."
                for i, w in enumerate(weights):
                    diffusers_to_bfl[f"{prefix}{w}"] = (i, key.replace("()", f"{b}"))
    for key, weights in BFL_TO_DIFFUSERS_MAP.items():
        if not (key.startswith("double") or key.startswith("single")):
            for i, w in enumerate(weights):
                diffusers_to_bfl[w] = (i, key)
    return diffusers_to_bfl

def convert_diffusers_sd_to_bfl(diffusers_sd, n_double=19, n_single=38):
    d_to_b = make_diffusers_to_bfl_map(n_double, n_single)
    flux_sd = {}
    for d_key, tensor in diffusers_sd.items():
        if d_key in d_to_b:
            idx, b_key = d_to_b[d_key]
            if b_key not in flux_sd: flux_sd[b_key] = []
            flux_sd[b_key].append((idx, tensor))
    
    for k, v in flux_sd.items():
        if len(v) == 1: flux_sd[k] = v[0][1]
        else: flux_sd[k] = torch.cat([x[1] for x in sorted(v, key=lambda y: y[0])])
    
    def swap_scale_shift(w):
        shift, scale = w.chunk(2, dim=0)
        return torch.cat([scale, shift], dim=0)
        
    if "final_layer.adaLN_modulation.1.weight" in flux_sd:
        flux_sd["final_layer.adaLN_modulation.1.weight"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.weight"])
    if "final_layer.adaLN_modulation.1.bias" in flux_sd:
        flux_sd["final_layer.adaLN_modulation.1.bias"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.bias"])
        
    return flux_sd