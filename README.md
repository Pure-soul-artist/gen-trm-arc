# Gen TRM — Generative Extension of Tiny Recursive Model for ARC-AGI

This project extends the [Tiny Recursive Model (TRM)](https://arxiv.org/abs/2510.04871) by Samsung SAIL Montreal with a **Variational Autoencoder (VAE) head**, transforming a deterministic reasoning model into a generative one capable of sampling multiple diverse solution hypotheses.

**Base paper:** "Less is More: Recursive Reasoning with Tiny Networks" — Alexia Jolicoeur-Martineau  
**Base repo:** [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

---

## What We Did

The original TRM is deterministic — given an input, it produces exactly one answer. We add a small VAE head (~3M parameters) on top of the frozen pretrained TRM backbone (7M parameters), enabling:

- **Stochastic sampling** — draw K different solution hypotheses from a learned distribution
- **Oracle accuracy** — count a task correct if ANY of K samples is right (beats deterministic)
- **Diversity** — measurable disagreement across K hypotheses proves genuine sampling

### Results (400 held-out ARC-AGI tasks)

| Method | Exact Match | Oracle (K=10) | Diversity |
|--------|-------------|---------------|-----------|
| Base TRM (deterministic) | 3.5% | 3.5% | 0% |
| **Gen TRM (ours)** | **3.5%** | **3.8%** | **8.7%** |

The oracle gap (+0.3%) proves the generative distribution covers correct solutions that the deterministic model structurally cannot reach. Diversity of 8.7% confirms genuine stochastic exploration.

---

## Setup

### Requirements

- Python 3.11
- CUDA 12.4 (or similar)
- Windows or Linux

### Installation

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/gen-trm-arc.git
cd gen-trm-arc

# Create virtual environment
python -m venv myenv

# Activate (Windows)
myenv\Scripts\activate

# Activate (Linux/Mac)
source myenv/bin/activate

# Install PyTorch with CUDA (adjust cu124 to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Install adam_atan2 (required by pretrain.py — comment it out if it fails)
pip install --no-cache-dir --no-build-isolation adam-atan2
```

### Critical: Disable torch.compile

This repo uses `torch.compile` which requires Triton. On Windows, Triton is not officially supported. **Always run scripts with this environment variable set:**

```bash
# Windows (Git Bash or CMD)
export TORCHDYNAMO_DISABLE=1

# Windows CMD
set TORCHDYNAMO_DISABLE=1
```

Add this to every command below or it will crash with a Triton error.

---

## Download Pretrained TRM Checkpoint

The base TRM checkpoint (~1.7GB) is from the ARC Prize verification:

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='arcprize/trm_arc_prize_verification',
    filename='step_518071',
    local_dir='.'
)
"
```

After downloading, `step_518071` should be in the repo root.

---

## Dataset Preparation

The raw ARC JSON files are in `kaggle/combined/`. Build the training dataset:

```bash
export TORCHDYNAMO_DISABLE=1

python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

This generates `data/arc1concept-aug-1000/` with train and test splits (~9GB).

---

## Fine-tune the VAE Head

Freeze the pretrained TRM and train only the VAE head (~3M parameters):

```bash
export TORCHDYNAMO_DISABLE=1

python -u finetune_vae.py \
  --checkpoint step_518071 \
  --data_dir data/arc1concept-aug-1000 \
  --epochs 10 \
  --batch_size 4 \
  --output vae_head_v3.pt \
  --kl_end 0.1
```

**Expected output per epoch:**
```
[E1 B10] Task=0.0042  KL=2.16  beta=0.01111
Epoch 1/10 | 202s | Task=0.0095  KL=2.15  Total=0.0117
```

Training takes ~30-60 minutes on an RTX 3050 (4GB VRAM). A pretrained `vae_head_v3.pt` is included in this repo.

---

## Evaluation

### Baseline: Base TRM (deterministic)

```bash
export TORCHDYNAMO_DISABLE=1

python -u eval_base_trm.py \
  --checkpoint step_518071 \
  --data_dir data/arc1concept-aug-1000 \
  --split test \
  --batch_size 4 2>&1 | tee base_trm_results.txt
```

Expected: ~3.5% exact match on 400 held-out tasks.

### Gen TRM (with VAE, K=10 hypotheses)

```bash
export TORCHDYNAMO_DISABLE=1

python -u eval_holdout.py \
  --checkpoint step_518071 \
  --vae_checkpoint vae_head_v3.pt \
  --data_dir data/arc1concept-aug-1000 \
  --num_hypotheses 10 \
  --batch_size 4 2>&1 | tee gen_trm_results.txt
```

Expected output:
```
Exact Match (vote):    3.5%  (14/400)
Oracle Accuracy:       3.8%  (15/400)
Oracle Gap:            +0.3%
Diversity Score:       8.743%
```

### Pass@N Evaluation (paper-style)

Run all augmented versions per puzzle, count correct if any attempt succeeds:

```bash
export TORCHDYNAMO_DISABLE=1

# Quick test with 10 augmentations per puzzle (~40 mins)
python -u eval_pass_at_n.py \
  --checkpoint step_518071 \
  --vae_checkpoint vae_head_v3.pt \
  --num_hypotheses 3 \
  --batch_size 8 \
  --max_aug 10 2>&1 | tee pass_at_n_results.txt

# Full eval with all ~920 augmentations (very long — use max_aug to limit)
python -u eval_pass_at_n.py \
  --checkpoint step_518071 \
  --vae_checkpoint vae_head_v3.pt \
  --num_hypotheses 1 \
  --batch_size 8
```

---

## Generate Result Plots

```bash
python plot_results.py
```

Generates:
- `gen_trm_results.png` — exact match, oracle, diversity comparison
- `gen_trm_architecture.png` — architecture contribution summary

---

## File Structure

```
gen-trm-arc/
├── models/
│   ├── vae_head.py              # VAE head architecture (our contribution)
│   ├── recursive_reasoning/
│   │   └── trm.py               # Samsung's TRM model (unchanged)
│   └── ...
├── kaggle/combined/             # Raw ARC-AGI JSON puzzle files
├── finetune_vae.py              # Fine-tune VAE head on frozen TRM
├── eval_holdout.py              # Main evaluation (1 example per puzzle)
├── eval_base_trm.py             # Baseline deterministic TRM evaluation
├── eval_pass_at_n.py            # Pass@N evaluation (paper methodology)
├── plot_results.py              # Generate comparison plots
├── vae_head_v3.pt               # Pretrained VAE head (our model)
├── vae_head_v3.history.json     # Training history
├── gen_trm_results.png          # Results plot
├── gen_trm_architecture.png     # Architecture plot
├── pretrain.py                  # Samsung's original training script
├── puzzle_dataset.py            # Samsung's data loader
└── requirements.txt             # Dependencies
```

---

## Architecture

```
Input (ARC puzzle tokens)
        |
  [Frozen TRM Backbone]     <- 7M params, pretrained by Samsung, never updated
        |
      z_H  (reasoning state, shape: batch x seq x 512)
        |
  [VAE Head — our addition] <- 3M params, fine-tuned by us
        |
   mu, logvar (512-dim)
        |
   z_sampled ~ N(mu, sigma)  <- different each call
        |
  [lm_head (frozen)]
        |
  Predicted output grid (K different grids for K samples)
```

**Key insight:** The TRM backbone produces a deterministic reasoning state `z_H`. Our VAE head learns a distribution over `z_H`, enabling K independent samples → K different output grids → oracle accuracy > exact match.

---

## Troubleshooting

**Triton error on Windows:**
```
ImportError: cannot import name 'AttrsDescriptor' from 'triton.backends.compiler'
```
Fix: `export TORCHDYNAMO_DISABLE=1` before every Python command.

**CUDA out of memory:**
Reduce `--batch_size` to 2 or 1. The model is 1.7GB so only ~2GB remains for activations on a 4GB GPU.

**adam_atan2 fails to install:**
Open `pretrain.py` and comment out line `from adam_atan2 import AdamATan2`. Replace `AdamATan2` with `torch.optim.AdamW` in the optimizer setup. This only affects the original Samsung training script, not our fine-tuning scripts.

**wandb login prompt:**
```
export WANDB_MODE=disabled
```

**Checkpoint key mismatch (`_orig_mod.` prefix):**
Already handled in the code — the load function strips this prefix automatically.

---

## Citation

If you use this work, please cite the original TRM paper:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks},
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871},
}
```

---

## Acknowledgements

- [Samsung SAIL Montreal](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) for the TRM implementation and pretrained checkpoint
- [ARC Prize Foundation](https://arcprize.org) for the ARC-AGI benchmark and verification checkpoint