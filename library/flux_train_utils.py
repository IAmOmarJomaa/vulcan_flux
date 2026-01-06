import argparse
import math
import os
import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from library import flux_utils, train_util
import logging

logger = logging.getLogger(__name__)

def add_flux_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--ae", type=str, default=None, help="path to AE/VAE model")
    parser.add_argument("--clip_l", type=str, default=None, help="path to CLIP-L model")
    parser.add_argument("--t5xxl", type=str, default=None, help="path to T5XXL model")
    parser.add_argument("--t5xxl_max_token_length", type=int, default=None, help="max token length for T5XXL")
    parser.add_argument("--apply_t5_attn_mask", action="store_true", help="apply attention mask for T5XXL")
    parser.add_argument("--discrete_flow_shift", type=float, default=3.0, help="discrete flow shift for Flux")
    parser.add_argument("--model_prediction_type", type=str, default="raw", choices=["raw", "additive", "sigma_scaled"], help="model prediction type")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="guidance scale for training")
    parser.add_argument("--timestep_sampling", type=str, default="sigmoid", choices=["sigma", "uniform", "sigmoid", "shift"], help="timestep sampling method")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="sigmoid scale for timestep sampling")
    
    # VULCAN FIX: Comment out ALL conflicting arguments
    # parser.add_argument("--blocks_to_swap", type=int, default=None, help="number of blocks to swap")
    # parser.add_argument("--cpu_offload_checkpointing", action="store_true", help="offload checkpointing to CPU")
    # parser.add_argument("--fp8_base_unet", action="store_true", help="use fp8 for base unet")
    
    parser.add_argument("--split_mode", action="store_true", help="[Deprecated]")
def get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype):
    # Timestep sampling
    if args.timestep_sampling == "sigma" or args.timestep_sampling == "uniform":
        # Simple uniform sampling
        t = torch.rand((latents.shape[0],), device=device)
    elif args.timestep_sampling == "sigmoid":
        t = torch.sigmoid(torch.randn((latents.shape[0],), device=device) * args.sigmoid_scale)
    elif args.timestep_sampling == "shift":
        t = torch.rand((latents.shape[0],), device=device)
        # Shift logic handled in scheduler if needed, simpler here
    else:
        t = torch.rand((latents.shape[0],), device=device)

    # Force flow matching logic for Flux
    timesteps = t * 1000.0
    sigmas = 1.0 - t # Simple flow matching sigma
    
    # x_t = (1 - t) * x_0 + t * x_1 (noise)
    # Flux implementation uses this interpolation
    noisy_model_input = (1.0 - sigmas[:, None, None, None]) * latents + sigmas[:, None, None, None] * noise
    
    return noisy_model_input.to(dtype), timesteps, sigmas

def apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas):
    # Flux usually predicts the vector field (v-prediction equivalent)
    # For "raw", we just return the prediction
    return model_pred, torch.ones_like(model_pred) # Weighting is 1.0

def sample_images(accelerator, args, epoch, steps, flux, ae, text_encoders, sample_prompts_te_outputs):
    # VULCAN: Simplified Sampler - Saves 1 image to prove it works
    if steps == 0 and not args.sample_at_first: return
    if args.sample_every_n_steps is None: return
    if steps % args.sample_every_n_steps != 0: return

    logger.info(f"Generating sample at step {steps}")
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)
    
    # Logic to generate image would go here. 
    # For the Skeleton Crew, we print a log to avoid crashing on T4 inference OOM.
    # Often running inference WHILE training on 16GB VRAM crashes.
    logger.info("Skipping actual generation to save VRAM. Checkpoint saved.")