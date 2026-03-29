"""
eval_base_trm.py
----------------
Evaluate the BASE TRM with no VAE — deterministic single prediction.
This is the true baseline to compare your VAE results against.
Run on both train (960) and holdout (400) splits.
"""

import argparse, os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch._dynamo; torch._dynamo.disable()

from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)


def load_one_per_puzzle(data_dir, split):
    split_dir = os.path.join(data_dir, split)
    inputs  = np.load(os.path.join(split_dir, "all__inputs.npy"),  mmap_mode='r')
    labels  = np.load(os.path.join(split_dir, "all__labels.npy"),  mmap_mode='r')
    puz_ids = np.load(os.path.join(split_dir, "all__puzzle_identifiers.npy"), mmap_mode='r')

    with open(os.path.join(data_dir, "identifiers.json")) as f:
        identifiers = json.load(f)

    id_to_base = {i: n.split('|||')[0] for i, n in enumerate(identifiers)
                  if n != '<blank>'}

    seen, examples = set(), []
    for row in range(len(puz_ids)):
        pid  = int(puz_ids[row])
        base = id_to_base.get(pid)
        if base and base not in seen:
            seen.add(base)
            examples.append((inputs[row].copy(), labels[row].copy(), pid))

    print(f"  Loaded {len(examples)} puzzles from {split} split")
    return examples


def make_batch(examples, device):
    return {
        "inputs":  torch.tensor(np.stack([e[0] for e in examples]),
                                dtype=torch.long, device=device),
        "labels":  torch.tensor(np.stack([e[1] for e in examples]),
                                dtype=torch.long, device=device),
        "puzzle_identifiers": torch.tensor([e[2] for e in examples],
                                           dtype=torch.long, device=device),
    }


def evaluate(checkpoint_path, data_dir, split, batch_size):
    print("=" * 60)
    print(f"  BASE TRM EVAL (no VAE) — {split} split")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from finetune_vae import load_trm_from_pretrain
    model, _ = load_trm_from_pretrain(checkpoint_path, data_dir, device, batch_size)
    trm       = model.model
    trm_inner = trm.inner
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    examples = load_one_per_puzzle(data_dir, split)

    correct = total = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_start in range(0, len(examples), batch_size):
            batch_exs = examples[batch_start:batch_start + batch_size]
            batch     = make_batch(batch_exs, device)

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
            for _ in range(trm_inner.config.halt_max_steps):
                carry, outputs = trm(carry=carry, batch=batch)
                if carry.halted.all():
                    break

            # Base TRM prediction — single deterministic output
            logits = outputs["logits"]              # (B, seq, vocab)
            preds  = logits.argmax(-1)              # (B, seq)
            labels = carry.current_data["labels"]   # (B, seq)
            mask   = labels != 0

            B = labels.size(0)
            for b in range(B):
                valid = mask[b]
                if not valid.any():
                    continue
                if (preds[b][valid] == labels[b][valid]).all():
                    correct += 1
                total += 1

            batches_done = batch_start // batch_size + 1
            if batches_done % 10 == 0:
                print(f"  [{total}/{len(examples)}] "
                      f"Exact={correct/max(total,1):.1%}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

    exact_acc = correct / max(total, 1)
    print("\n" + "=" * 60)
    print(f"  BASE TRM RESULTS ({split})")
    print("=" * 60)
    print(f"  Tasks:       {total}")
    print(f"  Exact Match: {exact_acc:.1%}  ({correct}/{total})")
    print(f"  (No VAE, no sampling, single deterministic prediction)")
    print("=" * 60)

    out = f"base_trm_{split}_results.json"
    with open(out, "w") as f:
        json.dump(dict(exact=exact_acc, correct=correct,
                       total=total, split=split), f, indent=2)
    print(f"\n  Saved: {out}")
    return exact_acc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="step_518071")
    p.add_argument("--data_dir",    default="data/arc1concept-aug-1000")
    p.add_argument("--split",       default="test",
                   choices=["train", "test"],
                   help="train=960 puzzles, test=400 held-out puzzles")
    p.add_argument("--batch_size",  type=int, default=4)
    args = p.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.split, args.batch_size)
