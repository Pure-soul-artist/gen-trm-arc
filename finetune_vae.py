"""
finetune_vae.py
---------------
Fine-tune only the VAE head on top of the frozen pretrained TRM.

What this does:
  1. Loads the pretrained TRM checkpoint (frozen — weights never change)
  2. Adds a small VAEHead (~3M new parameters)
  3. Trains ONLY the VAEHead for a few epochs on ARC data
  4. Saves the VAEHead weights separately

After training, use infer_vae.py to sample K hypotheses and evaluate.

Usage:
    python finetune_vae.py \
        --checkpoint step_518071 \
        --data_dir data/arc1concept-aug-1000 \
        --epochs 5 \
        --batch_size 8 \
        --output vae_head_trained.pt

The TRM checkpoint is ~1.8GB. The VAE head output is ~12MB.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── make sure repo root is on path ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vae_head import VAEHead, kl_loss
from models.losses import IGNORE_LABEL_ID

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

# ── load TRM (same logic as pretrain.py load_checkpoint) ──────────────────────
def load_frozen_trm(checkpoint_path: str, device: str):
    """
    Load TRM from checkpoint and freeze all weights.
    Returns the inner model (TinyRecursiveReasoningModel_ACTV1_Inner)
    since we need direct access to z_H.
    """
    from utils.functions import load_model_class

    # Build config matching the checkpoint (from all_config.yaml on HuggingFace)
    config_dict = dict(
        batch_size=1,        # placeholder, overridden per batch
        seq_len=1024,        # will be set from dataset metadata
        puzzle_emb_ndim=512,
        num_puzzle_identifiers=10000,  # placeholder
        vocab_size=33,       # ARC vocab
        H_cycles=3,
        L_cycles=4,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        expansion=4,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        forward_dtype="bfloat16",
        mlp_t=False,
        puzzle_emb_len=16,
        no_ACT_continue=True,
    )

    model_cls = load_model_class(
        "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1"
    )
    loss_cls  = load_model_class("losses@ACTLossHead")

    # We need dataset metadata to get the real seq_len and vocab_size
    # Load it from the data directory
    return config_dict, model_cls, loss_cls


def load_trm_from_pretrain(checkpoint_path: str, data_dir: str, device: str,
                            batch_size: int = 8):
    from omegaconf import OmegaConf
    from pretrain import PretrainConfig, create_dataloader, init_train_state

    cfg = OmegaConf.create({
        "arch": {
            "name": "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1",
            "loss": {
                "name": "losses@ACTLossHead",
                "loss_type": "stablemax_cross_entropy",
            },
            "halt_exploration_prob": 0.1,
            "halt_max_steps": 16,
            "H_cycles": 3,
            "L_cycles": 4,
            "H_layers": 0,
            "L_layers": 2,
            "hidden_size": 512,
            "expansion": 4,
            "num_heads": 8,
            "pos_encodings": "rope",
            "forward_dtype": "bfloat16",
            "mlp_t": False,
            "puzzle_emb_len": 16,
            "puzzle_emb_ndim": 512,
            "no_ACT_continue": True,
        },
        "data_paths": [data_dir],
        "data_paths_test": [],
        "evaluators": [{"name": "arc@ARC"}],
        "global_batch_size": batch_size,
        "epochs": 1,
        "eval_interval": 999999,
        "checkpoint_every_eval": False,
        "lr": 1e-4,
        "lr_min_ratio": 1.0,
        "lr_warmup_steps": 0,
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "puzzle_emb_lr": 1e-2,
        "puzzle_emb_weight_decay": 0.1,
        "seed": 0,
        "min_eval_interval": 0,
        "ema": False,
        "ema_rate": 0.999,
        "freeze_weights": False,
        "load_checkpoint": checkpoint_path,
        "checkpoint_path": None,
        "project_name": None,
        "run_name": "vae_finetune",
    })

    config = PretrainConfig(**OmegaConf.to_container(cfg, resolve=True))

    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    train_dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[data_dir],
        global_batch_size=batch_size,
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    ), split="train")
    train_loader   = torch.utils.data.DataLoader(train_dataset, batch_size=None)
    train_metadata = train_dataset.metadata

    train_state = init_train_state(config, train_metadata, rank=0, world_size=1)
    model = train_state.model
    return model, train_loader

# ── extract z_H from a forward pass ───────────────────────────────────────────
def get_z_H(trm_inner, carry, batch):
    """
    Run one forward pass through TRM inner model and return z_H
    without going through lm_head.

    trm_inner: TinyRecursiveReasoningModel_ACTV1_Inner
    Returns: z_H (batch, seq_len + puzzle_emb_len, hidden_size)
    """
    seq_info = dict(
        cos_sin=trm_inner.rotary_emb() if hasattr(trm_inner, "rotary_emb") else None,
    )
    input_embeddings = trm_inner._input_embeddings(
        batch["inputs"], batch["puzzle_identifiers"]
    )

    z_H, z_L = carry.z_H, carry.z_L

    with torch.no_grad():
        # Full deep recursion (all H_cycles, all with no_grad since TRM is frozen)
        for _H_step in range(trm_inner.config.H_cycles - 1):
            for _L_step in range(trm_inner.config.L_cycles):
                z_L = trm_inner.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = trm_inner.L_level(z_H, z_L, **seq_info)
        # Final pass also no_grad since TRM is frozen
        for _L_step in range(trm_inner.config.L_cycles):
            z_L = trm_inner.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = trm_inner.L_level(z_H, z_L, **seq_info)

    return z_H  # (batch, seq_len + puzzle_emb_len, hidden_size)


# ── training loop ─────────────────────────────────────────────────────────────
def train_vae(checkpoint_path, data_dir, epochs, batch_size, output_path,
              kl_weight_start=0.0, kl_weight_end=0.001, device="cuda"):

    print("=" * 60)
    print("  VAE Head Fine-tuning on Frozen TRM")
    print("=" * 60)
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Data:        {data_dir}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Output:      {output_path}")
    print(f"  Device:      {device}")

    # ── load frozen TRM ───────────────────────────────────────────────────────
    print("\nLoading frozen TRM...")
    model, train_loader = load_trm_from_pretrain(
        checkpoint_path, data_dir, device, batch_size
    )

    # Freeze the entire TRM — no gradients will flow into it
    trm_loss_head = model          # ACTLossHead
    trm           = trm_loss_head.model   # TinyRecursiveReasoningModel_ACTV1
    trm_inner     = trm.inner             # TinyRecursiveReasoningModel_ACTV1_Inner

    for param in trm_loss_head.parameters():
        param.requires_grad = False

    # Verify freeze worked
    assert not any(p.requires_grad for p in trm_loss_head.parameters()), \
        "TRM freeze failed!"

    frozen_params  = sum(p.numel() for p in trm_loss_head.parameters())
    print(f"  Frozen TRM parameters:  {frozen_params:,}")
    print(f"  All TRM requires_grad:  "
          f"{all(not p.requires_grad for p in trm_loss_head.parameters())}")

    # ── VAE head (the ONLY thing that trains) ─────────────────────────────────
    vae = VAEHead(hidden_size=512).to(device).to(torch.bfloat16)

    trainable = sum(p.numel() for p in vae.parameters())
    print(f"  Trainable VAE parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    history = dict(task_loss=[], kl_loss=[], total_loss=[])
    best_loss = float("inf")

    # ── training epochs ───────────────────────────────────────────────────────
    for epoch in range(epochs):
        t0 = time.time()

        # KL weight annealing — start at 0 so model learns task first
        kl_w = kl_weight_start + (kl_weight_end - kl_weight_start) * (epoch / max(epochs - 1, 1))

        running_task = running_kl = running_total = 0.0
        n_batches = 0

        for batch_idx, (_, batch, _) in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # ── get z_H from frozen TRM ───────────────────────────────────────
            # Initialise carry (reset for each new batch — fresh reasoning)
            # Move carry to device after initialising
            carry = trm.initial_carry(batch)
            carry = TinyRecursiveReasoningModel_ACTV1Carry(
                inner_carry=TinyRecursiveReasoningModel_ACTV1InnerCarry(
                    z_H=carry.inner_carry.z_H.cuda(),
                    z_L=carry.inner_carry.z_L.cuda(),
                ),
                steps=carry.steps.cuda(),
                halted=carry.halted.cuda(),
                current_data={k: v.cuda() if isinstance(v, torch.Tensor) else v
                            for k, v in carry.current_data.items()},
            )
            # Run Nsup=16 supervision steps to get a well-refined z_H
            # (same as what happens during normal TRM inference)
            with torch.no_grad():
                for _step in range(trm_inner.config.halt_max_steps):
                    carry, _ = trm(carry=carry, batch=batch)
                    if carry.halted.all():
                        break
            # Final z_H after full reasoning
            z_H = carry.inner_carry.z_H  # (B, seq_len+puz_len, 512) bfloat16

            # ── VAE forward pass ──────────────────────────────────────────────
            z_perturbed, mu, logvar = vae(z_H)
            # z_perturbed: (B, seq_len+puz_len, 512)

            # Decode using frozen lm_head
            with torch.no_grad():
                # Cast to same dtype as lm_head
                pass
            logits = trm_inner.lm_head(z_perturbed)  # (B, seq_len+puz_len, vocab)
            logits = logits[:, trm_inner.puzzle_emb_len:]  # strip puzzle prefix

            # ── loss ──────────────────────────────────────────────────────────
            labels = carry.current_data["labels"]  # (B, seq_len)

            # Task loss: stablemax cross entropy matching TRM's own loss
            from models.losses import stablemax_cross_entropy
            mask      = labels != IGNORE_LABEL_ID
            loss_div  = mask.sum(-1).clamp_min(1).unsqueeze(-1)
            task_loss = (stablemax_cross_entropy(
                logits.to(torch.float64), labels, ignore_index=IGNORE_LABEL_ID
            ) / loss_div).sum()

            kl = kl_loss(mu.float(), logvar.float(), free_bits=0.5)
            total_loss = task_loss + kl_w * kl

            # ── backward (only hits VAE parameters) ──────────────────────────
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            running_task  += task_loss.item()
            running_kl    += kl.item()
            running_total += total_loss.item()
            n_batches     += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  [E{epoch+1} B{batch_idx+1}] "
                      f"Task={task_loss.item():.4f}  "
                      f"KL={kl.item():.4f}  "
                      f"beta={kl_w:.5f}")

        scheduler.step()
        avg_task  = running_task  / max(n_batches, 1)
        avg_kl    = running_kl    / max(n_batches, 1)
        avg_total = running_total / max(n_batches, 1)

        print(f"\nEpoch {epoch+1}/{epochs} | {time.time()-t0:.0f}s | "
              f"Task={avg_task:.4f}  KL={avg_kl:.4f}  Total={avg_total:.4f}")

        history["task_loss"].append(avg_task)
        history["kl_loss"].append(avg_kl)
        history["total_loss"].append(avg_total)

        # Save best VAE head
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                "vae_state_dict": vae.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
                "hidden_size": 512,
            }, output_path)
            print(f"  Saved best VAE head -> {output_path}")

    # Save training history
    hist_path = Path(output_path).with_suffix(".history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved: {hist_path}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"VAE head saved: {output_path}")
    return vae


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.disable()

    p = argparse.ArgumentParser(description="Fine-tune VAE head on frozen TRM")
    p.add_argument("--checkpoint",  default="step_518071",
                   help="Path to pretrained TRM checkpoint")
    p.add_argument("--data_dir",    default="data/arc1concept-aug-1000",
                   help="ARC data directory")
    p.add_argument("--epochs",      type=int, default=5,
                   help="Number of fine-tuning epochs")
    p.add_argument("--batch_size",  type=int, default=8,
                   help="Batch size (keep small for 4GB VRAM)")
    p.add_argument("--output",      default="vae_head_trained.pt",
                   help="Output path for trained VAE head")
    p.add_argument("--kl_end",      type=float, default=0.001,
                   help="Final KL weight (annealed from 0)")
    args = p.parse_args()

    train_vae(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=args.output,
        kl_weight_end=args.kl_end,
    )
