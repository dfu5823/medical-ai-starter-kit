# Medical AI Starter Kit
## MD with Distinction in AI
## Session 1: Launching Your AI Research Workflow

Created: Dec 2025 
Last Modified: Dec 2025

Optional pre-session activity for **Session 1: Launching Your AI Research Workflow**. Run a quick experiment so you can follow along  during the live walkthrough on Jan 5th 2025.

No pressure: even opening the repo and skimming the structure is useful.

---

## Who this is for

- Medical students curious about AI health research workflows who want a concrete example
- People with basic coding experience (even minimal) but limited deep learning background
- Anyone who wants to see how real ML projects are organized, tracked, and compared

You do **not** need deep learning knowledge. The focus is on workflow and structure.

---

## Time + hardware expectations

### Time
- **Experienced with AI/ML setups:** ~5–10 minutes
- **Some Python, new to AI/ML tooling:** ~20–40 minutes
- **Very new / hit setup issues:** up to ~60 minutes is normal

Any amount of progress is fine — this is optional and meant as a reference for the live session.

### Hardware
- **CPU-only:** yes
- **Disk:** plan for ~30–50 MB (dataset + environment caches vary)
- On a Mac laptop, the baseline run took ~2 minutes (CPU-only)

> Tested on **Mac**. Should work on Windows/Linux, but setup steps (conda, shell scripts) may differ slightly.

---

## Why this matters

Many courses teach AI concepts but skip the practical skills needed to:
- set up projects cleanly
- run experiments reproducibly
- track results professionally
- compare runs without confusion

This repo is a small, realistic template you can reuse for future research projects.

---

## What you’ll learn

- How to run a full ML experiment from a config file
- How a research repo is organized (data → model → training → evaluation → visualization)
- How to track experiments using Weights & Biases (W&B)
- How to make a controlled change and see how results shift

---

## Assignment (simple)

**Goal:** Run **one** experiment, then change **one** variable in the config and rerun it.

### Step 0 — Install essentials (one-time)
- **VSCode:** https://code.visualstudio.com
- **Git:** https://git-scm.com/downloads
- **Conda:** https://docs.conda.io/en/latest/miniconda.html (Miniconda recommended)
- **Weights & Biases:** https://wandb.ai (free account, ~1 minute)

### Step 1 — Open VSCode + clone the repo
1. Open **VSCode**.
2. File → **Open Folder…** and choose where the project should live.
3. Open a terminal inside VSCode (Terminal → New Terminal) and run:
   ```bash
   git clone https://github.com/dfu5823/medical-ai-starter-kit
   cd medical-ai-starter-kit
   ```

### Step 2 — Create the environment
```bash
bash setup.sh
conda activate dermamnist-demo
```
This creates the Python environment and installs required packages.

### Step 3 — (Recommended) Login to Weights & Biases (only needs to be done once)
```bash
wandb login
```
If you don’t want W&B, set in `configs/baseline.yaml`:
```yaml
wandb:
  enabled: false
```

### Step 4 — Run the baseline experiment
```bash
python train.py --config configs/baseline.yaml
```
You should see training progress, figures in `./outputs/`, and (if enabled) a W&B link.

### Step 5 — Make your own config + change ONE knob
1. Copy the baseline config to a new file, e.g. `configs/my_experiment.yaml`. (Refer to sample_experiment.yaml for an example)
2. Change a single setting, then rerun:
   ```bash
   python train.py --config configs/my_experiment.yaml
   ```

Easy knobs to try:
- `seed` → different random result
- `aug.aug_strength` → `light` → `medium` → `chaos`
- `data.image_size` → `128` → `256` (higher resolution, slower)
- `model.freeze_backbone` → `true` → `false` (slower but interesting)
- `train.lr` → `1e-3` → `3e-4`
- `data.subset_train` → `2000` → `800` (faster)
- `train.epochs` → `2` → `5`

### Step 6 — (Optional) Turn on AI coding help in VSCode
- **GitHub Copilot:** VSCode → Extensions (Left Toolbar) → search “GitHub Copilot”
- **OpenAI/Codex-style tools (Cursor, etc.):** optional - also have VSCode extensions (I prefer Codex)

---

## What matters most

- Running it once is already a win.
- Changing one knob and seeing different results is the goal.
- It’s okay to get stuck — this repo is mainly a reference for the live session.

---

## What to prepare to share (optional)

If you ran it successfully, be ready to share one of:
1. Your W&B run link **or**
2. A screenshot of a confusion matrix, ROC plot, or “most confident wrong predictions” gallery

And be ready to say:
- what **one knob** you changed
- what you noticed changed (or didn’t)

Sharing is optional.

---

## Repo layout (how to navigate)

```text
medical-ai-starter-kit/
├── train.py                        # single entry point (runs the full pipeline)
├── Session_1_Launching_Your_AI_Research_Workflow.py
├── configs/                        # experiment configs (what you edit)
│   └── baseline.yaml
├── src/                            # modular implementation
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── logging/
│   ├── models/
│   ├── pipeline/
│   ├── training/
│   ├── utils/
│   └── viz/
├── data/                           # downloaded datasets (gitignored)
├── outputs/                        # figures and artifacts (gitignored)
├── wandb/                          # local W&B run logs (gitignored)
├── environment.yml
└── setup.sh
```

If you’re new to repos, start with:
1. `train.py`
2. `configs/baseline.yaml`
3. `src/pipeline/run.py`

---

## Expected outputs

**Saved locally in `./outputs/`:**
- training curves (loss/accuracy)
- confusion matrix
- ROC curves (one-vs-rest, 7 classes)
- sample images by class
- “most confident wrong predictions” gallery

**In W&B (if enabled):**
- metrics (accuracy, macro-F1, macro AUC, per-class AUC)
- plots and image artifacts
- easy comparisons between runs after you change one knob

---

## Troubleshooting + common fallbacks

### 1) “It’s too slow”
- `data.subset_train: 800`
- `data.subset_val: 200`
- `train.epochs: 1`
- `data.image_size: 96`
- `train.batch_size: 32`

### 2) “I’m running out of memory / crashes”
- reduce `train.batch_size` (e.g., 64 → 32 → 16)
- reduce `data.image_size` (128 → 96)
- reduce subsets

### 3) “My storage is limited”
- change dataset location:
  ```yaml
  data:
    root: "/path/to/external_drive/data"
  ```
- delete the downloaded dataset in `data/` if needed (it can be re-downloaded)
- conda envs can be large; consider cleaning:
  ```bash
  conda clean --all
  ```

### 4) “W&B isn’t working / I don’t want to make an account”
- disable it:
  ```yaml
  wandb:
    enabled: false
  ```
- or offline mode:
  ```yaml
  wandb:
    mode: offline
  ```

### 5) Dependency issues (most common)
- confirm you activated the env:
  ```bash
  conda activate dermamnist-demo
  ```
- sanity check imports:
  ```bash
  python -c "import torch, torchvision, medmnist, sklearn, yaml; print('ok')"
  ```

### 6) Common warnings (usually safe to ignore)
- PIL EXIF warnings, torchvision warnings, etc.
- “CUDA not available” (expected on CPU)
- “num_workers=0” (expected on portable setup)

### 7) “Wrong config file / typo in YAML”
- start from `configs/baseline.yaml` and make one change at a time
- watch indentation carefully (YAML is indentation-sensitive)

### 8) “I edited the wrong file”
- only edit files under `configs/` for this assignment
- avoid changing `src/` unless you’re comfortable

### 9) Bad usage patterns (avoid)
- changing 5 knobs at once (hard to attribute changes)
- setting `freeze_backbone: false` with high epochs on CPU (too slow)
- increasing `image_size` a lot (training gets much slower)

---

## If you get stuck

That’s normal. Aim for any of these checkpoints:
- repo cloned and opened in VSCode
- conda env created and activated
- dataset downloaded
- one run started

Use ChatGPT or AI Coding tools from Step 6 to get unstuck! Simply copy-paste the error you are having or just describe it and ask it to help explain to you.

Even partial progress gives you a concrete reference during the live session.

Good Luck!