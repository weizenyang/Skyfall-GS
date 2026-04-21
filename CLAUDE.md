# Skyfall-GS runbook for Claude

Self-contained guide for installing and running this repo. Read top to bottom — the ordering matters.

## What this is

Fork of the paper [Skyfall-GS](https://arxiv.org/abs/2510.15869): synthesizing explorable 3D city scenes from satellite imagery via a two-stage pipeline (Mip-Splatting reconstruction → FLUX.1-Dev-driven iterative dataset update). Upstream is `jayin92/Skyfall-GS`; this fork is `weizenyang/Skyfall-GS`.

## Fixes already applied in this fork (don't redo)

Compared to upstream `jayin92/Skyfall-GS@main`, this fork already contains:

1. **`scene/gaussian_model.py::load_ply`** — upstream has all six parameter assignments (`_xyz`/features/opacity/scaling/rotation) commented out, so `load_ply` only reads `filter_3D`. Here they are uncommented. Breaks `Scene` resume-from-iteration and `create_fused_ply.py` without this fix.
2. **`create_fused_ply.py`** — rewritten to:
   - Build a `Scene` (gets training cameras + invokes `load_ply`)
   - Optionally load checkpoint (restores appearance MLP + embeddings)
   - **Recompute `filter_3D` via `compute_3D_filter(trainCameras)`** against the final `_xyz` (upstream uses stale `filter_3D` read from the saved PLY, causing "large sparse Gaussians" in the fused output)
   - Accept a separate `--color_mapped` flag (upstream conflates it with `--load_from_checkpoints`)
3. **`.gitmodules`** — URLs fixed. Three previously pointed at unpublished `weizenyang/*` forks causing "repository does not seem to exist" errors.

Do not revert any of these. If you need to check upstream behavior, `git remote -v` shows `upstream → jayin92/Skyfall-GS`.

## Prerequisites

- Windows or Linux with NVIDIA GPU (≥24 GB VRAM recommended — FLUX.1-Dev is the bottleneck in Stage 2)
- Conda (Anaconda/Miniconda)
- `gh` CLI authenticated (only needed if publishing changes to the user's forks)
- HuggingFace account with access to the gated `black-forest-labs/FLUX.1-dev` repo (accept license on the HF page first)

## Installation

```bash
git clone --recurse-submodules https://github.com/weizenyang/Skyfall-GS.git
cd Skyfall-GS
conda create -y -n skyfall-gs python=3.10
conda activate skyfall-gs
conda install -y cuda-toolkit=12.8 cuda-nvcc=12.8 -c nvidia
pip install -r requirements.txt
pip install --force-reinstall torch torchvision torchaudio
pip install submodules/diff-gaussian-rasterization-depth submodules/simple-knn submodules/fused-ssim
huggingface-cli login
```

After `huggingface-cli login`, visit https://huggingface.co/black-forest-labs/FLUX.1-dev and click "Agree and access repository" — FLUX is gated, Stage 2 will hang at model load without this.

### Optional: download the paper's reference datasets

```bash
huggingface-cli download jayinnn/Skyfall-GS-datasets --repo-type dataset --local-dir ./data/
```

Unzip any `.zip` files so paths end up like `./data/datasets_JAX/JAX_068/`.

## Dataset format

A scene directory must contain at minimum:

```
data/<scene>/
  transforms_train.json       # required — triggers "Satellite" scene loader
  points3D.txt                # required (alt: depths_moge/)
  transforms_test.json        # optional
  images/                     # RGB images referenced by transforms
  masks/                      # optional
```

`scene/__init__.py:46-49` routes to the `Satellite` loader when `transforms_train.json + points3D.txt` are both present. Otherwise it tries Blender/Colmap/Multi-scale in that order.

## Running the pipeline

Assumes your scene is at `./data/my_scene`. Substitute as needed.

### Stage 1 — Reconstruction (~1h 35min on A6000, per paper)

```bash
python train.py -s ./data/my_scene -m ./outputs/my_scene --eval --port 6209 --kernel_size 0.1 --resolution 1 --sh_degree 1 --appearance_enabled --lambda_depth 0 --lambda_opacity 10 --densify_until_iter 21000 --densify_grad_threshold 0.0001 --lambda_pseudo_depth 0.5 --start_sample_pseudo 1000 --end_sample_pseudo 21000 --size_threshold 20 --scaling_lr 0.001 --rotation_lr 0.001 --opacity_reset_interval 3000 --sample_pseudo_interval 10
```

Produces `./outputs/my_scene/chkpnt30000.pth` and `./outputs/my_scene/point_cloud/iteration_30000/point_cloud.ply`.

### Stage 2 — IDU with FLUX.1-Dev (~5h 10min on A6000, per paper)

```bash
python train.py -s ./data/my_scene -m ./outputs/my_scene_idu --start_checkpoint ./outputs/my_scene/chkpnt30000.pth --iterative_datasets_update --eval --port 6209 --kernel_size 0.1 --resolution 1 --sh_degree 1 --appearance_enabled --lambda_depth 0 --lambda_opacity 0 --idu_opacity_reset_interval 5000 --idu_refine --idu_num_samples_per_view 2 --densify_grad_threshold 0.0002 --idu_num_cams 6 --idu_use_flow_edit --idu_render_size 1024 --idu_flow_edit_n_min 4 --idu_flow_edit_n_max 10 --idu_grid_size 3 --idu_grid_width 512 --idu_grid_height 512 --idu_episode_iterations 10000 --idu_iter_full_train 0 --idu_opacity_cooling_iterations 500 --lambda_pseudo_depth 0.5 --idu_densify_until_iter 9000 --idu_train_ratio 0.75
```

Produces `./outputs/my_scene_idu/chkpnt80000.pth`.

### Stage 3 — Fused PLY export

```bash
python create_fused_ply.py -m ./outputs/my_scene_idu --output_ply ./fused/my_scene_fused.ply --iteration 80000 --load_from_checkpoints
```

Append `--color_mapped` to apply the appearance-MLP tone-mapping (uses the embedding at uid=6; may darken colors since DC SH coefs get `clamp_max(1.0)`). Without it you get raw SH colors — usually what you want for a viewer.

## Verifying Stage 1 is healthy mid-training

Watch the `# of GS` counter in the progress bar. If after an opacity reset (every 3000 iters) the count collapses toward zero and never recovers, the scene has too few views for the default `--lambda_opacity 10`. Kill the run and check the saved PLYs:

```python
from plyfile import PlyData
import os
base = './outputs/<scene>/point_cloud'
for d in sorted(os.listdir(base), key=lambda x: int(x.split('_')[1])):
    p = PlyData.read(os.path.join(base, d, 'point_cloud.ply'))
    print(f'{d}: {len(p.elements[0].data)} vertices')
```

If iteration_20000 onward shows 0 vertices, reduce pressure. The paper's `lambda_opacity=10` is tuned for dense satellite coverage. For sparse scenes try `--lambda_opacity 0 --opacity_reset_interval 100000` (disables the BCE-entropy push and the periodic reset), but only after confirming the dense setting actually fails on your scene.

## Known gotchas

- **FlowEdit submodule empty after clone**: if `submodules/FlowEdit/` has no files, re-run `git submodule update --init --recursive`. Stage 2 imports `FlowEditRefineIDU` at module load time and crashes immediately if this is empty.
- **`create_fused_ply.py` requires `--load_from_checkpoints`** when the model was trained with `--appearance_enabled`. Appearance MLP + embeddings live only in `.pth`, not the PLY.
- **Empty checkpoint silently continues**: if Stage 1 collapsed to 0 Gaussians, Stage 2 will happily FLUX-refine blank renders for hours without error. Check Stage 1 vertex count before kicking off Stage 2.
- **FLUX.1-Dev download is ~24 GB** the first run — don't assume the hang at `Loading pipeline components` is a bug; it's the initial model pull.

## File map

| Path | Purpose |
|---|---|
| `train.py` | Both stages. Path chosen by `--iterative_datasets_update` flag |
| `create_fused_ply.py` | Post-training export (already fixed) |
| `scene/__init__.py` | Dataset loader dispatcher |
| `scene/gaussian_model.py` | Gaussian params + densification + `load_ply`/`save_fused_ply` (`load_ply` already fixed) |
| `gaussian_renderer/__init__.py` | Forward render; applies 3D filter + 2D kernel |
| `submodules/FlowEdit/idu_refine.py` | FLUX wrapper, consumes `--idu_flow_edit_n_min/max` |
| `submodules/diff-gaussian-rasterization-depth/` | Custom rasterizer (our fork with Windows USE_CUDA fix at SHA `89a47fc`) |
| `scripts/run_{jax,nyc}{,_idu,_naive}.py` | Batch wrappers for the JAX/NYC datasets |

## When the user reports "bad output"

Likely causes in order of prevalence:

1. Empty checkpoint from Stage 1 collapse → check vertex counts (see above)
2. Used raw `point_cloud.ply` from training instead of the fused one → always use `create_fused_ply.py` for viewers
3. Stale `filter_3D` → already handled by our `create_fused_ply.py` rewrite; verify it still calls `compute_3D_filter` before `save_fused_ply`
4. Viewer doesn't understand Mip-Splatting's format → fused PLY is meant to be viewer-compatible; if not, check that `--color_mapped` wasn't silently on
