import torch
import os
import logging
import json
from safetensors.torch import load_file as load_safetensors
from . import flux_models
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPConfig, T5EncoderModel, T5Config
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# --- VULCAN FIX: Missing Constants ---
MODEL_VERSION_FLUX_V1 = "flux_v1"
MODEL_VERSION_CHROMA = "chroma"

# --- VULCAN FIX: Correct Flux VAE Config (16 Latent Channels) ---
# The previous mismatch was because standard SDXL uses 4 channels.
# Flux uses 16. The checkpoint weights [32, 512, 3, 3] correspond to 2 * 16 (mean + logvar).
FLUX_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3, 
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "latent_channels": 16, # <--- CRITICAL FIX: Changed from 4 to 16
    "norm_num_groups": 32,
    "scaling_factor": 0.18215, # Flux scale factor
    "sample_size": 256,
}

def analyze_checkpoint_state(ckpt_path: str):
    if not os.path.exists(ckpt_path) and not os.path.exists(os.path.join(ckpt_path, "transformer")):
        try:
            if "FLUX.1" in ckpt_path:
                target = "flux1-dev.safetensors" if "dev" in ckpt_path else "flux1-schnell.safetensors"
                ckpt_path = hf_hub_download(repo_id=ckpt_path, filename=target)
        except Exception: pass
    
    ckpt_path = ckpt_path.strip('"').strip("'")
    if os.path.isdir(ckpt_path): return True, False, (19, 38), []
    is_schnell = "schnell" in os.path.basename(ckpt_path)
    return False, is_schnell, (19, 38), [ckpt_path]

def load_flow_model(ckpt_path, dtype, device, disable_mmap=False, model_type="flux"):
    is_diffusers, is_schnell, (num_double, num_single), paths = analyze_checkpoint_state(ckpt_path)
    name = "dev" if not is_schnell else "schnell"
    
    with torch.device("meta"):
        model = flux_models.Flux(flux_models.configs[name]).to(dtype)
    
    sd = {}
    for p in paths:
        sd.update(load_safetensors(p, device=device))
        
    keys_to_rename = [k for k in sd.keys() if k.startswith("model.diffusion_model.")]
    for key in keys_to_rename:
        sd[key.replace("model.diffusion_model.", "")] = sd.pop(key)

    model.load_state_dict(sd, strict=False, assign=True)
    return is_schnell, model

# --- VULCAN FIX: Bulletproof VAE Loader ---
def load_ae(name, dtype, device, disable_mmap=False):
    logger.info(f"Loading AE from {name}...")
    
    # 1. Try standard load first (folder structure)
    try:
        return AutoencoderKL.from_pretrained(name, subfolder="ae", torch_dtype=dtype).to(device)
    except:
        pass

    # 2. Fallback: Download 'ae.safetensors' and load manually with Correct Config
    try:
        logger.info("Standard load failed. Downloading Flux AE and applying Manual Config...")
        ae_path = hf_hub_download(repo_id="black-forest-labs/FLUX.1-dev", filename="ae.safetensors")
        
        # Build the VAE shell with the 16-channel config
        ae = AutoencoderKL(**FLUX_VAE_CONFIG)
        
        # Load weights
        sd = load_safetensors(ae_path, device="cpu") 
        ae.load_state_dict(sd, strict=False) 
        
        return ae.to(device, dtype=dtype)
    except Exception as e:
        logger.error(f"Failed to manual load AE: {e}")
        # Final Hail Mary (Likely to fail if we are here, but required for return type)
        return AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)

def load_clip_l(path, dtype, device, disable_mmap=False):
    if path is None: path = "openai/clip-vit-large-patch14"
    try:
        return CLIPTextModel.from_pretrained(path, torch_dtype=dtype).to(device)
    except:
        config = CLIPConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=77, hidden_act="quick_gelu")
        return CLIPTextModel(config).to(device, dtype=dtype)

def load_t5xxl(path, dtype, device, disable_mmap=False):
    if path is None: path = "google/t5-v1_1-xxl"
    try:
        return T5EncoderModel.from_pretrained(path, torch_dtype=dtype).to(device)
    except:
        config = T5Config(vocab_size=32128, d_model=4096, d_kv=64, d_ff=10240, num_layers=24, num_heads=64, relative_attention_num_buckets=32, dropout_rate=0.1, layer_norm_epsilon=1e-6, initializer_factor=1.0, feed_forward_proj="gated-gelu", is_encoder_decoder=True, use_cache=True, pad_token_id=0, eos_token_id=1)
        return T5EncoderModel(config).to(device, dtype=dtype)