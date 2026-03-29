"""
eval_holdout.py
---------------
Evaluate VAE head on held-out ARC data — one augmented example per
original training puzzle, from the test split of arc1concept-aug-1000.

This gives ~400 unseen examples with correct puzzle embeddings,
finishing in ~15 minutes. Legitimate held-out evaluation.

Usage:
    python eval_holdout.py \
        --checkpoint step_518071 \
        --vae_checkpoint vae_head_v3.pt \
        --num_hypotheses 10
"""

import argparse, os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch._dynamo; torch._dynamo.disable()

from models.vae_head import VAEHead
from models.losses import IGNORE_LABEL_ID
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)


def load_one_per_puzzle(data_dir: str):
    """
    Load exactly one held-out example per original puzzle.
    Groups augmented variants by base puzzle name (part before |||)
    and picks the first example of each original puzzle.
    """
    split_dir = os.path.join(data_dir, "test")
    inputs  = np.load(os.path.join(split_dir, "all__inputs.npy"),
                      mmap_mode='r')
    labels  = np.load(os.path.join(split_dir, "all__labels.npy"),
                      mmap_mode='r')
    puz_ids = np.load(os.path.join(split_dir, "all__puzzle_identifiers.npy"),
                      mmap_mode='r')

    # Build map: identifier_index -> original puzzle name
    with open(os.path.join(data_dir, "identifiers.json")) as f:
        identifiers = json.load(f)

    # Map each identifier ID to its base puzzle name
    id_to_base = {}
    for i, name in enumerate(identifiers):
        if name == '<blank>':
            continue
        base = name.split('|||')[0]
        id_to_base[i] = base

    # Pick first example per original puzzle name
    seen_bases = set()
    examples   = []
    for row in range(len(puz_ids)):
        pid  = int(puz_ids[row])
        base = id_to_base.get(pid)
        if base is None:
            continue
        if base not in seen_bases:
            seen_bases.add(base)
            examples.append((
                inputs[row].copy(),
                labels[row].copy(),
                pid,
            ))

    print(f"  Selected {len(examples)} unique original puzzles")
    return examples

def make_batch(examples, device):
    inputs  = torch.tensor(np.stack([e[0] for e in examples]),
                           dtype=torch.long, device=device)
    labels  = torch.tensor(np.stack([e[1] for e in examples]),
                           dtype=torch.long, device=device)
    puz_ids = torch.tensor([e[2] for e in examples],
                           dtype=torch.long, device=device)
    return {"inputs": inputs, "labels": labels,
            "puzzle_identifiers": puz_ids}


def evaluate(checkpoint_path, vae_checkpoint_path, data_dir,
             num_hypotheses=10, batch_size=4):

    print("=" * 60)
    print("  Held-out ARC Evaluation (1 example per puzzle)")
    print(f"  K = {num_hypotheses} hypotheses per task")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── load frozen TRM ───────────────────────────────────────────────────────
    print("\nLoading TRM...")
    from finetune_vae import load_trm_from_pretrain
    model, _ = load_trm_from_pretrain(
        checkpoint_path, data_dir, device, batch_size
    )
    trm       = model.model
    trm_inner = trm.inner

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # ── load VAE head ─────────────────────────────────────────────────────────
    print(f"Loading VAE head from {vae_checkpoint_path}...")
    ckpt = torch.load(vae_checkpoint_path, map_location=device)
    vae  = VAEHead(hidden_size=ckpt.get("hidden_size", 512))
    vae.load_state_dict(ckpt["vae_state_dict"], strict=False)
    vae  = vae.to(device).to(torch.bfloat16)
    vae.eval()
    print(f"  Epoch {ckpt.get('epoch','?')}  "
          f"loss={ckpt.get('loss',0):.4f}  "
          f"noise_scale={vae.noise_scale}")

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading held-out examples from {data_dir}...")
    examples = load_one_per_puzzle(data_dir)

    # ── evaluation ────────────────────────────────────────────────────────────
    exact_correct   = 0
    oracle_correct  = 0
    total_tasks     = 0
    total_diversity = 0.0
    t0 = time.time()

    print(f"\nEvaluating {len(examples)} puzzles in batches of {batch_size}...")

    with torch.no_grad():
        for batch_start in range(0, len(examples), batch_size):
            batch_exs = examples[batch_start:batch_start + batch_size]
            batch     = make_batch(batch_exs, device=device)

            # Run TRM to convergence
            carry = trm.initial_carry(batch)
            carry = TinyRecursiveReasoningModel_ACTV1Carry(
                inner_carry=TinyRecursiveReasoningModel_ACTV1InnerCarry(
                    z_H=carry.inner_carry.z_H.to(device),
                    z_L=carry.inner_carry.z_L.to(device),
                ),
                steps=carry.steps.to(device),
                halted=carry.halted.to(device),
                current_data={
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in carry.current_data.items()
                },
            )
            for _step in range(trm_inner.config.halt_max_steps):
                carry, _ = trm(carry=carry, batch=batch)
                if carry.halted.all():
                    break

            z_H    = carry.inner_carry.z_H
            labels = carry.current_data["labels"]
            mask   = labels != 0   # 0 = padding

            # K hypothesis samples
            all_z, mu, logvar = vae.sample_n(z_H, num_hypotheses)
            all_preds = []
            for k in range(num_hypotheses):
                logits_k = trm_inner.lm_head(all_z[k])
                logits_k = logits_k[:, trm_inner.puzzle_emb_len:]
                all_preds.append(logits_k.argmax(-1))
            all_preds = torch.stack(all_preds, dim=0)   # (K, B, seq)

            # Per-task metrics
            B = labels.size(0)
            for b in range(B):
                valid = mask[b]
                if not valid.any():
                    continue
                gt = labels[b]

                # Majority vote
                vote_pred = all_preds[:, b, :].mode(dim=0).values
                if (vote_pred[valid] == gt[valid]).all():
                    exact_correct += 1

                # Oracle
                for k in range(num_hypotheses):
                    if (all_preds[k, b][valid] == gt[valid]).all():
                        oracle_correct += 1
                        break

                # Diversity
                preds_b  = all_preds[:, b, :]
                disagree = ~(preds_b == preds_b[0].unsqueeze(0)).all(0)
                total_diversity += disagree[valid].float().mean().item()

                total_tasks += 1

            batches_done = batch_start // batch_size + 1
            total_batches = (len(examples) + batch_size - 1) // batch_size
            if batches_done % 10 == 0:
                print(f"  [{total_tasks}/{len(examples)}] "
                      f"Exact={exact_correct/max(total_tasks,1):.1%}  "
                      f"Oracle={oracle_correct/max(total_tasks,1):.1%}  "
                      f"({time.time()-t0:.0f}s)",
                      flush=True)

    # ── results ───────────────────────────────────────────────────────────────
    exact_acc  = exact_correct  / max(total_tasks, 1)
    oracle_acc = oracle_correct / max(total_tasks, 1)
    diversity  = total_diversity / max(total_tasks, 1)

    print("\n" + "=" * 60)
    print("  HELD-OUT ARC RESULTS")
    print("=" * 60)
    print(f"  Tasks evaluated:       {total_tasks}")
    print(f"  Hypotheses (K):        {num_hypotheses}")
    print(f"  Time:                  {time.time()-t0:.0f}s")
    print()
    print(f"  Exact Match (vote):    {exact_acc:.1%}  ({exact_correct}/{total_tasks})")
    print(f"  Oracle Accuracy:       {oracle_acc:.1%}  ({oracle_correct}/{total_tasks})")
    print(f"  Oracle Gap:            +{(oracle_acc - exact_acc):.1%}")
    print(f"  Diversity Score:       {diversity:.3%}")
    print()
    print(f"  Paper baseline (TRM):  ~45.0%")
    print(f"  Your model exact:      {exact_acc:.1%}")
    print(f"  Your model oracle:     {oracle_acc:.1%}")
    print("=" * 60)

    results = dict(
        exact_match=exact_acc,
        oracle_accuracy=oracle_acc,
        oracle_gap=oracle_acc - exact_acc,
        diversity=diversity,
        total_tasks=total_tasks,
        K=num_hypotheses,
        note="1 held-out augmented example per original training puzzle"
    )
    with open("vae_holdout_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: vae_holdout_results.json")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default="step_518071")
    p.add_argument("--vae_checkpoint", default="vae_head_v3.pt")
    p.add_argument("--data_dir",       default="data/arc1concept-aug-1000")
    p.add_argument("--num_hypotheses", type=int, default=10)
    p.add_argument("--batch_size",     type=int, default=4)
    args = p.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        vae_checkpoint_path=args.vae_checkpoint,
        data_dir=args.data_dir,
        num_hypotheses=args.num_hypotheses,
        batch_size=args.batch_size,
    )
