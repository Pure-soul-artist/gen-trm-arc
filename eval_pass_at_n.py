"""
eval_pass_at_n.py
-----------------
Evaluate Gen TRM using the same methodology as the paper:
  - Run ALL augmented versions of each test puzzle
  - Count puzzle correct if ANY attempt succeeds
  - This is "pass@920" matching the paper's ~45% evaluation

Also runs Base TRM the same way for direct comparison.

This is the apples-to-apples comparison with the paper's 45%.

Usage:
    python eval_pass_at_n.py \
        --checkpoint step_518071 \
        --vae_checkpoint vae_head_v3.pt \
        --num_hypotheses 1
"""

import argparse, os, sys, json, time
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch._dynamo; torch._dynamo.disable()

from models.vae_head import VAEHead
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)


def load_all_augmented(data_dir: str):
    """
    Load ALL augmented examples from test split, grouped by original puzzle.
    Returns dict: {base_puzzle_name: [(input, label, pid), ...]}
    """
    split_dir = os.path.join(data_dir, "test")
    inputs  = np.load(os.path.join(split_dir, "all__inputs.npy"),  mmap_mode='r')
    labels  = np.load(os.path.join(split_dir, "all__labels.npy"),  mmap_mode='r')
    puz_ids = np.load(os.path.join(split_dir, "all__puzzle_identifiers.npy"), mmap_mode='r')

    with open(os.path.join(data_dir, "identifiers.json")) as f:
        identifiers = json.load(f)

    id_to_base = {i: n.split('|||')[0] for i, n in enumerate(identifiers)
                  if n != '<blank>'}

    puzzles = defaultdict(list)
    for row in range(len(puz_ids)):
        pid  = int(puz_ids[row])
        base = id_to_base.get(pid)
        if base:
            puzzles[base].append((row, pid))

    print(f"  Loaded {len(puzzles)} original puzzles")
    print(f"  Total augmented examples: {sum(len(v) for v in puzzles.values())}")
    avg = sum(len(v) for v in puzzles.values()) / len(puzzles)
    print(f"  Avg augmentations per puzzle: {avg:.0f}")

    return puzzles, inputs, labels


def make_batch(rows, inputs, labels, puz_ids_list, device):
    inp = torch.tensor(np.stack([inputs[r] for r in rows]),
                       dtype=torch.long, device=device)
    lbl = torch.tensor(np.stack([labels[r] for r in rows]),
                       dtype=torch.long, device=device)
    pid = torch.tensor(puz_ids_list, dtype=torch.long, device=device)
    return {"inputs": inp, "labels": lbl, "puzzle_identifiers": pid}


def run_batch(trm, trm_inner, batch, device):
    """Run TRM on batch, return final outputs and carry."""
    carry = trm.initial_carry(batch)
    carry = TinyRecursiveReasoningModel_ACTV1Carry(
        inner_carry=TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=carry.inner_carry.z_H.to(device),
            z_L=carry.inner_carry.z_L.to(device),
        ),
        steps=carry.steps.to(device),
        halted=carry.halted.to(device),
        current_data={k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in carry.current_data.items()},
    )
    outputs = None
    for _ in range(trm_inner.config.halt_max_steps):
        carry, outputs = trm(carry=carry, batch=batch)
        if carry.halted.all():
            break
    return carry, outputs


def evaluate(checkpoint_path, vae_checkpoint_path, data_dir,
             num_hypotheses=1, batch_size=8, max_aug=None):

    print("=" * 60)
    print("  PASS@N EVAL — Same methodology as paper")
    print(f"  Gen TRM hypotheses per augmentation: {num_hypotheses}")
    print(f"  Max augmentations per puzzle: {max_aug or 'all'}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── load TRM ──────────────────────────────────────────────────────────────
    print("\nLoading TRM...")
    from finetune_vae import load_trm_from_pretrain
    model, _ = load_trm_from_pretrain(checkpoint_path, data_dir, device, batch_size)
    trm       = model.model
    trm_inner = trm.inner
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # ── load VAE ──────────────────────────────────────────────────────────────
    use_vae = vae_checkpoint_path is not None
    if use_vae:
        print(f"Loading VAE head from {vae_checkpoint_path}...")
        ckpt = torch.load(vae_checkpoint_path, map_location=device)
        vae  = VAEHead(hidden_size=ckpt.get("hidden_size", 512))
        vae.load_state_dict(ckpt["vae_state_dict"], strict=False)
        vae  = vae.to(device).to(torch.bfloat16)
        vae.eval()
        print(f"  noise_scale={vae.noise_scale}")
    else:
        print("No VAE — running Base TRM only")

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading augmented test data...")
    puzzles, all_inputs, all_labels = load_all_augmented(data_dir)

    # ── evaluation ────────────────────────────────────────────────────────────
    base_correct = 0    # base TRM any correct
    gen_correct  = 0    # gen TRM any correct (if VAE enabled)
    total_puzzles = 0
    t0 = time.time()

    puzzle_list = list(puzzles.items())
    print(f"\nEvaluating {len(puzzle_list)} puzzles...")

    with torch.no_grad():
        for puzzle_idx, (base_name, aug_list) in enumerate(puzzle_list):

            # Optionally limit augmentations per puzzle
            if max_aug:
                aug_list = aug_list[:max_aug]

            rows    = [r for r, pid in aug_list]
            pid_list = [pid for r, pid in aug_list]

            base_any_correct = False
            gen_any_correct  = False

            # Process in batches
            for batch_start in range(0, len(rows), batch_size):
                batch_rows = rows[batch_start:batch_start + batch_size]
                batch_pids = pid_list[batch_start:batch_start + batch_size]
                batch = make_batch(batch_rows, all_inputs, all_labels,
                                   batch_pids, device)

                carry, outputs = run_batch(trm, trm_inner, batch, device)

                labels = carry.current_data["labels"]
                mask   = labels != 0
                B      = labels.size(0)

                # ── Base TRM check ────────────────────────────────────────────
                base_preds = outputs["logits"].argmax(-1)  # (B, seq)
                for b in range(B):
                    valid = mask[b]
                    if not valid.any():
                        continue
                    if (base_preds[b][valid] == labels[b][valid]).all():
                        base_any_correct = True

                # ── Gen TRM check ─────────────────────────────────────────────
                if use_vae and not gen_any_correct:
                    z_H = carry.inner_carry.z_H
                    all_z, _, _ = vae.sample_n(z_H, num_hypotheses)
                    for k in range(num_hypotheses):
                        logits_k = trm_inner.lm_head(all_z[k])
                        logits_k = logits_k[:, trm_inner.puzzle_emb_len:]
                        preds_k  = logits_k.argmax(-1)
                        for b in range(B):
                            valid = mask[b]
                            if not valid.any():
                                continue
                            if (preds_k[b][valid] == labels[b][valid]).all():
                                gen_any_correct = True

                if base_any_correct and (gen_any_correct or not use_vae):
                    break  # early exit — both already correct

            if base_any_correct:
                base_correct += 1
            if use_vae and gen_any_correct:
                gen_correct += 1
            total_puzzles += 1

            if (puzzle_idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (puzzle_idx + 1) * (len(puzzle_list) - puzzle_idx - 1)
                print(f"  [{total_puzzles}/{len(puzzle_list)}] "
                      f"Base={base_correct/total_puzzles:.1%}  "
                      f"Gen={gen_correct/total_puzzles:.1%}  "
                      f"({elapsed:.0f}s, ETA {eta:.0f}s)",
                      flush=True)

    base_acc = base_correct / max(total_puzzles, 1)
    gen_acc  = gen_correct  / max(total_puzzles, 1) if use_vae else 0

    print("\n" + "=" * 60)
    print("  PASS@N RESULTS")
    print("=" * 60)
    print(f"  Puzzles evaluated:     {total_puzzles}")
    print(f"  Avg augmentations:     ~{sum(len(v) for v in puzzles.values())//len(puzzles)}")
    print(f"  Gen hypotheses/aug:    {num_hypotheses}")
    print()
    print(f"  Base TRM (pass@920):   {base_acc:.1%}  ({base_correct}/{total_puzzles})")
    if use_vae:
        print(f"  Gen TRM  (pass@920):   {gen_acc:.1%}  ({gen_correct}/{total_puzzles})")
        print(f"  Gen improvement:       +{(gen_acc-base_acc):.1%}")
    print()
    print(f"  Paper baseline:        ~45%")
    print("=" * 60)

    results = dict(
        base_trm_pass_at_n=base_acc,
        gen_trm_pass_at_n=gen_acc if use_vae else None,
        improvement=gen_acc - base_acc if use_vae else None,
        total_puzzles=total_puzzles,
        avg_augmentations=sum(len(v) for v in puzzles.values())//len(puzzles),
        gen_hypotheses_per_aug=num_hypotheses,
    )
    with open("pass_at_n_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: pass_at_n_results.json")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      default="step_518071")
    p.add_argument("--vae_checkpoint",  default="vae_head_v3.pt")
    p.add_argument("--data_dir",        default="data/arc1concept-aug-1000")
    p.add_argument("--num_hypotheses",  type=int, default=1,
                   help="VAE samples per augmented example")
    p.add_argument("--batch_size",      type=int, default=8)
    p.add_argument("--max_aug",         type=int, default=None,
                   help="Limit augmentations per puzzle (None=all ~920)")
    args = p.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        vae_checkpoint_path=args.vae_checkpoint,
        data_dir=args.data_dir,
        num_hypotheses=args.num_hypotheses,
        batch_size=args.batch_size,
        max_aug=args.max_aug,
    )
