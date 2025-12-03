#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read optimizer_states from a NeMo / Megatron torch_dist distributed checkpoint.

This script reproduces the exact logic of:
    Strategy.load_checkpoint() when use_distributed_checkpointing=True

Specifically, the Strategy builds a checkpoint template like:

    sharded_state_dict = model.sharded_state_dict()
    checkpoint = {
        "state_dict": sharded_state_dict,
        "optimizer_states": [optimizer_sharded_state_dict(is_loading=True)]
    }

Then it calls:
    checkpoint_io.load_checkpoint(checkpoint_path, sharded_state_dict=checkpoint)

This script does the same thing:
  - Build the same template
  - Use torch.distributed.checkpoint (DCP) to fill the template
  - Extract optimizer_states for analysis (e.g., SVD / normalization)
"""

import os
import argparse
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp

from omegaconf import OmegaConf
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel


# ===================== Distributed Init =====================

def init_dist() -> None:
    """Initialize a minimal distributed environment (world_size = 1)."""
    if dist.is_initialized():
        return

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )


# ===================== Build Model + Optimizer =====================

def build_model_and_optimizer(cfg_path: str) -> Tuple[Any, torch.optim.Optimizer, Any]:
    """
    Build MegatronGPTModel and its optimizer based on the NeMo config.

    We only need the correct structure of model & optimizer to construct
    sharded_state_dict and optimizer_sharded_state_dict.
    """
    cfg = OmegaConf.load(cfg_path)

    # If your NeMo fork requires a trainer, pass a minimal trainer object.
    model = MegatronGPTModel(cfg.model, trainer=None)

    opt_conf = model.configure_optimizers()

    # NeMo usually returns: ([optimizers], [schedulers]) or a dict
    if isinstance(opt_conf, (list, tuple)):
        optimizers = opt_conf[0]
    elif isinstance(opt_conf, dict) and "optimizer" in opt_conf:
        optimizers = opt_conf["optimizer"]
    else:
        raise RuntimeError(f"Unexpected configure_optimizers return type: {type(opt_conf)}")

    if isinstance(optimizers, (list, tuple)):
        optimizer = optimizers[0]
    else:
        optimizer = optimizers

    return model, optimizer, cfg


# ===================== Build Checkpoint Template =====================

def build_checkpoint_template(model: Any, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    """
    Build a checkpoint template identical to NeMo's Strategy.load_checkpoint logic:

        sharded_state_dict = model.sharded_state_dict()
        checkpoint = {
            "state_dict": sharded_state_dict,
            "optimizer_states": [model.optimizer_sharded_state_dict(is_loading=True)]
        }

    This template will later be filled by torch.distributed.checkpoint.
    """
    # ----- Model state (prefer sharded_state_dict) -----
    if hasattr(model, "sharded_state_dict"):
        print("[INFO] Using model.sharded_state_dict() as template for 'state_dict'.")
        model_state = model.sharded_state_dict()
    else:
        print("[WARN] model.sharded_state_dict() not found. Falling back to model.state_dict().")
        model_state = model.state_dict()

    # ----- Optimizer state (prefer NeMo's sharded optimizer state) -----
    optim_state = None
    if hasattr(model, "optimizer_sharded_state_dict"):
        try:
            print("[INFO] Using model.optimizer_sharded_state_dict(is_loading=True) template.")
            optim_state = model.optimizer_sharded_state_dict(is_loading=True)
        except TypeError:
            print("[WARN] optimizer_sharded_state_dict signature mismatch. Retrying without is_loading.")
            optim_state = model.optimizer_sharded_state_dict()
        except Exception as e:
            print(f"[WARN] optimizer_sharded_state_dict failed: {e}. Falling back to optimizer.state_dict().")
            optim_state = None

    if optim_state is None:
        print("[WARN] Using optimizer.state_dict() as template for optimizer_states[0].")
        optim_state = optimizer.state_dict()

    checkpoint = {
        "state_dict": model_state,
        "optimizer_states": [optim_state],
    }

    print("[DEBUG] Checkpoint template keys:", list(checkpoint.keys()))
    return checkpoint


# ===================== Load from torch_dist Distributed Checkpoint =====================

def load_from_torchdist(checkpoint_template: Dict[str, Any], ckpt_dir: str) -> Dict[str, Any]:
    """
    Use torch.distributed.checkpoint (FileSystemReader) to fill the template dictionary.

    This replicates:
        checkpoint = checkpoint_io.load_checkpoint(checkpoint_path, sharded_state_dict=checkpoint)
    but using raw DCP APIs instead of Lightning's Strategy.
    """
    print(f"[INFO] Loading torch_dist checkpoint from directory: {ckpt_dir}")
    reader = dist_cp.FileSystemReader(ckpt_dir)

    # PyTorch 2.1+ supports load_state_dict; older versions use load()
    if hasattr(dist_cp, "load_state_dict"):
        dist_cp.load_state_dict(
            state_dict=checkpoint_template,
            storage_reader=reader,
        )
    else:
        dist_cp.load(
            state_dict=checkpoint_template,
            storage_reader=reader,
        )

    print("[INFO] torch_dist checkpoint successfully loaded into template.")
    print("[DEBUG] Post-load template keys:", list(checkpoint_template.keys()))
    return checkpoint_template


# ===================== Inspect / Modify Optimizer States =====================

def inspect_and_optionally_modify_optimizer_states(ckpt: Dict[str, Any]) -> None:
    """
    Inspect optimizer states and provide a place to apply post-processing (e.g., SVD).

    Two valid forms (matching NeMo's _integrate_original_checkpoint_data):

    Case A:
        optimizer_states[0] = {
            "optimizer": {
                "state": {...},
                "param_groups": [...],
                ...
            }
        }

    Case B:
        optimizer_states[0] = {
            "state": {...},
            "param_groups": [...],
            ...
        }
    """
    if "optimizer_states" not in ckpt or len(ckpt["optimizer_states"]) == 0:
        print("[ERROR] No 'optimizer_states' key found in checkpoint.")
        return

    opt_state_0 = ckpt["optimizer_states"][0]
    print("[DEBUG] optimizer_states[0] keys:", list(opt_state_0.keys()))

    # Match NeMo logic exactly
    if "optimizer" in opt_state_0:
        opt_inner = opt_state_0["optimizer"]
    else:
        opt_inner = opt_state_0

    state = opt_inner.get("state", {})
    param_groups = opt_inner.get("param_groups", [])

    print(f"[INFO] Number of param groups: {len(param_groups)}")
    print(f"[INFO] Number of parameters with optimizer state: {len(state)}")

    # Find a parameter that has exp_avg and exp_avg_sq
    sample_pid = None
    for pid, s in state.items():
        if isinstance(s, dict) and "exp_avg" in s and "exp_avg_sq" in s:
            sample_pid = pid
            break

    if sample_pid is None:
        print("[WARN] No parameter found with both exp_avg and exp_avg_sq.")
        return

    s = state[sample_pid]
    m = s["exp_avg"]
    v = s["exp_avg_sq"]

    print(f"[INFO] Sample param id: {sample_pid}")
    print(f"       exp_avg shape: {m.shape}, exp_avg_sq shape: {v.shape}, dtype: {m.dtype}")

    # ===== INSERT YOUR PROCESSING LOGIC HERE =====
    # Example:
    # eps = 1e-6
    # norm = m.norm() + eps
    # s["exp_avg"] = m / norm
    #
    # After modifying state, you may want to save it back.
    # =============================================


# ===================== Save optimizer_states to file =====================

def save_optimizer_states(ckpt: Dict[str, Any], save_path: str) -> None:
    """
    Save optimizer_states to a standalone .pt file for offline analysis.
    """
    if "optimizer_states" not in ckpt or len(ckpt["optimizer_states"]) == 0:
        raise RuntimeError("optimizer_states not found; nothing to save.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(ckpt["optimizer_states"], save_path)
    print(f"[INFO] Saved optimizer_states to: {save_path}")


# ===================== CLI & main =====================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read optimizer_states from a NeMo torch_dist distributed checkpoint "
                    "(keys match Strategy.load_checkpoint exactly)."
    )
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to the NeMo Megatron GPT training config YAML.")
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Path to the torch_dist checkpoint directory "
                             "(contains metadata.json and *.distcp files).")
    parser.add_argument("--save-optimizer", type=str, default=None,
                        help="Optional: save checkpoint['optimizer_states'] to a .pt file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    init_dist()

    # 1) Build model & optimizer
    model, optimizer, cfg = build_model_and_optimizer(args.cfg)

    # 2) Build template identical to NeMo Strategy.load_checkpoint
    ckpt_template = build_checkpoint_template(model, optimizer)

    # 3) Fill template using torch_dist
    ckpt_filled = load_from_torchdist(ckpt_template, args.ckpt_dir)

    # 4) Inspect / modify optimizer states
    inspect_and_optionally_modify_optimizer_states(ckpt_filled)

    # 5) Optional dump to .pt
    if args.save_optimizer is not None:
        save_optimizer_states(ckpt_filled, args.save_optimizer)


if __name__ == "__main__":
    main()
