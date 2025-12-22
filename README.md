````markdown
# DermMNIST Workflow Demo (CPU-Friendly) — MD with Distinction in AI Session 1

This repo is an **optional** pre-session activity for **Session 1: Launching Your AI Research Workflow**.  
It’s designed to help you **feel comfortable** during the live walkthrough and (if you want) to **shine** by having your own experiment run to reference and compare with others.

**No pressure:** it’s totally fine to try and get stuck, or to only complete part of it. Even opening the repo + config and seeing how it’s structured is useful.

---

## Who this is for

This is for medical students who:
- are curious about AI research workflows and want a concrete example
- have basic coding experience (even minimal) but limited deep learning experience
- want to learn how real ML projects are organized and tracked (e.g., Weights & Biases)

You do **not** need deep learning knowledge to do this. The goal is workflow + structure.

---

## Time + hardware expectations

### Time

- **If you’ve set up AI/ML research environments before:**  
  ~5–10 minutes start to finish
- **If you’ve done some Python but are new to AI/ML tooling:**  
  ~20–40 minutes (first-time setup, minor fixes)
- **If you’re very new or hit setup issues:**  
  Up to ~60 minutes is normal

Any amount of progress is fine — this is optional and meant as a reference for the live session.

### Hardware
- **CPU-only:** yes
- **Disk:** plan for **~30-50 MB** (dataset + environment caches vary)
- **On my laptop:** the “fast” run took **~2 minutes** (CPU-only)

> This repo has been tested on **Mac**. It should work on Windows/Linux too, but you may hit setup differences (especially with conda and shell scripts). That’s okay.

---

## Why this matters

In medical school, people often learn AI “concepts” but not the practical skills required to actually:
- set up a project cleanly
- run experiments reproducibly
- track results professionally
- compare runs, avoid confusion, and collaborate with others

This repo is a **small, realistic template** you can reuse for future research projects.

---

## What you’ll learn

By doing (or attempting) this pre-assignment, you’ll practice:

- **How to run a full ML experiment from a config file**
- **How a research repo is organized** (data → model → training → evaluation → visualization)
- **How to track experiments using Weights & Biases (W&B)** for comparisons and sharing
- **How to make a controlled change** (one variable) and see how experimental results change

---

## Assignment (simple)

**Goal:** Run **one** experiment, then change **one** variable in the config and rerun.

Here’s a **slightly more guided version** of the assignment that helps complete beginners without slowing down experienced folks. It adds just enough hand-holding at the right moments and keeps everything concise.

---

## Assignment (simple)

**Goal:** Run **one** experiment, then change **one** variable in the config and rerun it.

---

### Step 0 — Install essentials (one-time)

You’ll need:

* **VSCode (code editor)**
  [https://code.visualstudio.com](https://code.visualstudio.com)
  → download, install, open it once

* **Git (to clone the repo)**
  [https://git-scm.com/downloads](https://git-scm.com/downloads)
  → install with default settings

* **Conda (Python + environment manager)**
  [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
  → install Miniconda (recommended)

* **Weights & Biases (experiment tracking)**
  [https://wandb.ai](https://wandb.ai)
  → free account, takes ~1 minute

> If you’ve done any coding before, you likely already have some of these.

---

### Step 1 — Open VSCode + clone the repo

1. Open **VSCode**
2. Choose **File → Open Folder…**
3. Pick or create a folder on your computer where you want this project to live
   (e.g. `Documents/AI/`)
4. Open the **Terminal** inside VSCode:

   * Top menu → **Terminal → New Terminal**

In the terminal, run:

```bash
git clone <YOUR_REPO_URL>
cd dermamnist-workflow-demo
```

You should now see the project files in the VSCode file explorer.

---

### Step 2 — Create the environment

In the VSCode terminal:

```bash
bash setup.sh
```

This will:

* create a Python environment
* install all required packages

When it finishes, activate the environment if needed:

```bash
conda activate dermamnist-demo
```

---

### Step 3 — (Recommended) Login to Weights & Biases

```bash
wandb login
```

This lets you view results in your browser.

If you **don’t** want to use W&B, open `configs/baseline.yaml` and set:

```yaml
wandb:
  enabled: false
```

---

### Step 4 — Run the baseline experiment

In the terminal:

```bash
python train.py --config configs/baseline.yaml
```

You should see:

* training progress printed in the terminal
* figures saved to `./outputs/`
* (if W&B enabled) a link printed to your W&B dashboard

Let it finish — on CPU this should take only a few minutes.

---

### Step 5 — Make your own config + change ONE knob

Instead of editing the baseline file directly:

1. Copy it:

   * Right-click `configs/baseline.yaml`
   * Duplicate it
2. Rename the copy (for example):

   * `configs/my_experiment.yaml`

Open your new file and change **one** thing.

**Easy / fun knobs to try:**

* `seed` → different random result
* `aug.aug_strength` → `light` → `medium` → `chaos`
* `data.image_size` → `128` → `256` (higher resolution, slower)
* `model.freeze_backbone` → `true` → `false` *(slower but interesting)*
* `train.lr` → `1e-3` → `3e-4`
* `data.subset_train` → `2000` → `800` (faster)
* `train.epochs` → `2` → `5`

Then rerun **using your new config**:

```bash
python train.py --config configs/my_experiment.yaml
```

In W&B, you’ll now see **two runs** that you can compare side-by-side.

---

### Step 6 — (Optional) Turn on AI coding help in VSCode

If you want:

* **GitHub Copilot**
  VSCode → Extensions → search “GitHub Copilot” → enable
* **OpenAI / Codex-style tools** (Cursor, etc.)
  Optional; not required for this assignment

These can help explain files or answer questions, but you don’t need them.

---

### What matters most

* Running it once is already a win
* Changing one knob and seeing different results is the goal
* It’s okay to get stuck — this repo is mainly a **reference** for the live session

Even just opening the repo and understanding how it’s organized will help you follow along and compare during the workshop.

---

## What to prepare to share (optional)

If you ran it successfully, please be ready to share **one** of the following in the live session:

1. Your W&B run link **OR**
2. A screenshot of:

   * confusion matrix, or
   * ROC plot, or
   * “most confident wrong predictions” gallery

And be ready to say:

* what **one knob** you changed
* what you noticed changed (or didn’t)

**No pressure to present.** Sharing is optional.

---

## Repo layout (how to navigate)

```text
dermamnist-workflow-demo/
├── train.py                      # single entry point (runs the full pipeline)
├── configs/                       # experiment configs (what you edit)
│   ├── fast.yaml
│   ├── baseline.yaml
│   └── chaos_aug.yaml
├── src/                           # modular implementation
│   ├── config/                    # defaults + config loading/merge
│   ├── utils/                     # seed, device, paths
│   ├── logging/                   # W&B logger
│   ├── data/                      # dataset + transforms + loaders
│   ├── models/                    # ResNet backbone + head
│   ├── training/                  # train/validate loops + optimizer
│   ├── evaluation/                # prediction + metrics
│   ├── viz/                       # plots + image galleries
│   └── pipeline/                  # run_experiment orchestrator
├── outputs/                       # figures saved here (gitignored)
├── environment.yml
└── setup.sh
```

**If you’re new to repos:** start with:

1. `train.py`
2. `configs/fast.yaml`
3. `src/pipeline/run.py`

---

## Expected outputs

After a run, you should see:

**Saved locally in `./outputs/`:**

* training curves (loss/accuracy)
* confusion matrix
* ROC curves (one-vs-rest, 7 classes)
* sample images by class
* “most confident wrong predictions” gallery

**In W&B (if enabled):**

* metrics (accuracy, macro-F1, macro AUC, per-class AUC)
* plots and image artifacts
* easy comparisons between runs after you change one knob

---

## Troubleshooting + common fallbacks

### 1) “It’s too slow”

Try these config changes:

* `data.subset_train: 800`
* `data.subset_val: 200`
* `train.epochs: 1`
* `data.image_size: 96`
* `train.batch_size: 32`

### 2) “I’m running out of memory / crashes”

* reduce `train.batch_size` (e.g., 64 → 32 → 16)
* reduce `data.image_size` (128 → 96)
* reduce subsets

### 3) “My storage is limited”

* change dataset location:

  ```yaml
  data:
    root: "/path/to/external_drive/data"
  ```
* delete downloaded dataset in `data/` if needed (it can be re-downloaded)
* conda envs can be large; consider cleaning:

  ```bash
  conda clean --all
  ```

### 4) “W&B isn’t working / I don’t want to make an account”

No problem — disable it:

```yaml
wandb:
  enabled: false
```

Or offline mode:

```yaml
wandb:
  mode: offline
```

### 5) Dependency issues (most common)

* Confirm you activated the env:

  ```bash
  conda activate dermamnist-demo
  ```
* Sanity check imports:

  ```bash
  python -c "import torch, torchvision, medmnist, sklearn, yaml; print('ok')"
  ```

### 6) Common warnings (usually safe to ignore)

* PIL EXIF warnings, torchvision warnings, etc.
* “CUDA not available” (expected on CPU)
* “num_workers=0” (expected on portable setup)

### 7) “Wrong config file / typo in YAML”

Symptoms:

* KeyError (missing fields)
* YAML parse errors

Fix:

* Start from `configs/fast.yaml` and make one change at a time
* Watch indentation carefully (YAML is indentation-sensitive)

### 8) “I edited the wrong file”

Best practice:

* Only edit files under `configs/` for this assignment
* Don’t change `src/` unless you’re comfortable

### 9) Bad usage patterns (avoid)

* Changing 5 knobs at once (you won’t know what caused the change)
* Setting `freeze_backbone: false` + high epochs on CPU (too slow)
* Increasing `image_size` a lot (training gets much slower)

---

## If you get stuck

That’s normal.

If you have time, try to get to **any** of these checkpoints:

* repo cloned + opened in VSCode
* conda env created and activated
* dataset downloaded
* one run started (even if you stop it)

Even partial progress gives you a concrete reference during the live session.

---

## One-line summary

**Optional pre-session repo to run a CPU-friendly DermMNIST classification experiment, log results to W&B, and learn the “real workflow” behind AI research projects.**