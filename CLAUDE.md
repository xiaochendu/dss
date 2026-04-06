# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DSS** (Diffusion Structure Search) is a PyTorch Lightning research package for generating novel atomic surface structures using generative diffusion models. It focuses on AgxOy (silver oxide) surfaces and supports two generative objectives:
- **VP-Diffusion** (score matching, mode=`"diffusion"`)
- **Flow Matching** (OT-based velocity field, mode=`"flow_matching"`)

The project is a research prototype; a more user-friendly successor exists at [agedi](https://github.com/nronne/agedi).

## Installation

```bash
conda env create -f environment.yml
conda activate dss
pip install -e .
```

**Critical dependency note**: ASE must be pinned to `<3.26` for schnetpack compatibility. `mace-torch` pulls a newer ASE, so `environment.yml` explicitly re-pins it after install.

## Runner System (vssr_fm_working)

All experiments are managed through `/mnt/data0/dux/vssr_fm_working/runner/`. This is the primary interface for running and monitoring both DSS and snowy-flow-dev experiments.

```bash
cd /mnt/data0/dux/vssr_fm_working

# Launch an experiment (background, survives shell exit)
python runner/launch.py experiments/dss/001_diffusion_fixed_comp.yaml
python runner/launch.py snowy_001               # by ID

# Check progress
python runner/status.py                         # table
python runner/status.py --json                  # machine-readable (for Claude)
python runner/status.py --log dss_001           # tail log for one experiment
python runner/status.py --refresh 30            # auto-refresh

# Start the persistent watchdog (auto-restarts crashed runs)
bash runner/start_watchdog.sh                   # nohup, survives terminal close
tail -f logs/watchdog.log

# Grid sampling after training
python runner/sample.py dss_001
python runner/sample.py snowy_001 --num_samples 200

# Compare experiments
python runner/compare.py dss_001 snowy_001
```

**Experiment configs**: `experiments/{dss,snowy}/*.yaml` — one file per experiment, contains all hyperparameters, paths, and sampling config.
**Registry**: `experiments/registry.json` — live state of all runs (PID, run_dir, best_checkpoint, W&B ID, last_epoch).
**Outputs**: `outputs/agxoy/{dss,snowy}/{exp_id}/{timestamp}/`
**Samples**: `samples/agxoy/{exp_id}/Ag{x}O{y}/sample.xyz`
**Logs**: `logs/{exp_id}.log`

**`composition_conditioning` semantics (snowy)**:
- `composition_conditioning=true` → fixed composition (positions-only flow, DSS-style)
- `composition_conditioning=false` → variable composition (discrete + position flow)

**Phase 1** experiments (fixed comp): `dss_001`, `dss_002`, `snowy_001`, `snowy_003`
**Phase 2** experiments (variable comp): `snowy_002`

## Common Commands (direct invocation)

Run scripts with the `dss` conda environment and `PYTHONPATH` set:

```bash
conda activate dss
export PYTHONPATH=$PYTHONPATH:/home/dux/dss
```

There is no test suite or linter configured in this repository.

### Training — Fixed Composition Diffusion (VP)

```bash
/home/dux/miniforge3/envs/dss/bin/python /home/dux/dss/scripts/train.py \
--config $HOME/dss/config/train_agxoy.yaml \
--data_path /mnt/data0/dux/vssr_fm_working/data/agox_AgxOy_structures \
--mcmc_files agox_sample_AgxOy_2000.xyz \
--max_epochs 100 \
--limit_val_batches 5 \
--batch_size 32 \
--val_batch_size 32 \
--n_atom_basis 64 --n_interactions 3 --n_rbf 20 \
--surface_eval \
--val_sample_num_steps 100 \
--val_sample_postrelax_steps 0 \
--val_save_trajectories \
--check_val_every_n_epoch 5 \
--mace_model="medium" \
--mace_device="cuda:0" \
--mace_dtype="float32" \
--no_mace_dispersion \
--train_energies_path="/mnt/data0/dux/vssr_fm_working/data/agox_AgxOy_structures/agox_sample_AgxOy_2000_train_energies.pt" \
--wandb
```

### Training — Fixed Composition Flow Matching

```bash
/home/dux/miniforge3/envs/dss/bin/python /home/dux/dss/scripts/train.py \
--config $HOME/dss/config/train_agxoy_fm.yaml \
--data_path /mnt/data0/dux/vssr_fm_working/data/agox_AgxOy_structures \
--mcmc_files agox_sample_AgxOy_2000.xyz \
--max_epochs 100 \
--limit_val_batches 5 \
--batch_size 32 \
--val_batch_size 32 \
--n_atom_basis 64 --n_interactions 3 --n_rbf 20 \
--surface_eval \
--val_sample_num_steps 100 \
--val_sample_postrelax_steps 0 \
--val_save_trajectories \
--check_val_every_n_epoch 5 \
--mace_model="medium" \
--mace_device="cuda:0" \
--mace_dtype="float32" \
--no_mace_dispersion \
--train_energies_path="/mnt/data0/dux/vssr_fm_working/data/agox_AgxOy_structures/agox_sample_AgxOy_2000_train_energies.pt" \
--wandb
```

### Grid Sampling

```bash
python /home/dux/dss/scripts/grid_sample.py \
--checkpoint /mnt/data0/dux/vssr_fm_working/outputs/agxoy/dss_nf64_nr20_bs32_lr0.001/2026-03-07_23-00-59/checkpoints/24-val_loss-0.7023.ckpt \
--template /mnt/data0/dux/vssr_fm_working/data/agox_AgxOy_structures/template_111_c4x8.xyz \
--out_dir /mnt/data0/dux/vssr_fm_working/benchmarks/outputs/agxoy/fresh_sampling/dss \
--ag_min 4 --ag_max 6 \
--o_min 2 --o_max 5 \
--num_samples 100 \
--n_atom_basis 64 --n_interactions 3 --n_rbf 20 \
--z_conf_min 2.8613568 \
--z_conf_max 6.49999996
```

Output is written as `out_dir/Ag{x}O{y}/sample.xyz` in extended XYZ format.

## Architecture

### Data Flow

```
Raw XYZ data (MCMC samples + template slab)
  → AgxOy dataset loader (dss/data/agxoy.py)
  → schnetpack AtomsDataModule batches with mask, z_confinement, forces, energy
  → ESSFlow training module (dss/diffusion/vp.py)
  → SurfaceEvalCallback at validation (dss/callbacks/surface_eval.py)
```

### Core Modules

**`dss/diffusion/vp.py` — `ESSFlow`** (central training/sampling module)
- Implements both generative modes in a single `pl.LightningModule`
- `mode="diffusion"`: trains score network ∇log p(x_t|x_0); reverse process via SDE
- `mode="flow_matching"`: trains velocity field x₁−x₀ with OT assignment; reverse via ODE
- `sample()`: generates structures from a template + composition symbols list
- `regressor_guidance_sample()`: optional energy-guided sampling

**`dss/models/score.py` — `ConditionedScoreModel`**
- PaiNN (SchNetPack) backbone with time conditioning (sin/cos embeddings) and optional energy conditioning
- Outputs per-atom 3D score/velocity vectors via a gated equivariant MLP
- All conditioning is additive into atom features before message passing

**`dss/models/potential.py` — `Potential`**
- Thin wrapper around SchNetPack `NeuralNetworkPotential`
- Used optionally in the loss (energy/force weighting) and for validation-time energy evaluation

**`dss/energy/mace.py` — `MACEEnergyModel`**
- Lazy-loads the pre-trained MACE-MP foundation model
- Supports D3 dispersion correction and cuEquivariance acceleration
- Used by `SurfaceEvalCallback` when `--mace_model` is specified

**`dss/callbacks/surface_eval.py` — `SurfaceEvalCallback`**
- Runs at `on_validation_epoch_end`
- Precomputes training set energies once, then at each validation epoch: samples structures, evaluates energies (MACE or potential), and logs Wasserstein distance, composition distribution, and surface stability metrics
- Optionally saves XYZ trajectories and matplotlib plots

**`dss/utils/`** — GPU-accelerated neighbor list (`neighbourlist.py`), EMA (`ema.py`), truncated normal sampler for z-confinement (`truncated_normal.py`), PBC offset computation (`offsets.py`), OT assignment for flow matching (`ot.py`)

### Batch Format

Batches are SchNetPack-style dictionaries with keys:
- `_positions`, `_atomic_numbers`, `_cell`, `_pbc`, `_n_atoms`, `_idx_m`: structure
- `mask`: boolean, marks fixed (slab) vs. mobile (adsorbate) atoms — diffusion only acts on `mask==True`
- `z_confinement`: [z_min, z_max] bounding box for the adsorbate layer
- `energy`, `forces`: optional regression targets

### Configuration System

Training uses a simple YAML + CLI override pattern (no Hydra). A YAML config sets defaults; any CLI argument overrides it. Run directories are templated with `{param_name}` placeholders and `{now:%Y%m%d_%H%M%S}`.

Pre-built configs:
- `config/train_agxoy.yaml`: VP-Diffusion defaults (no MACE, loss-based checkpointing)
- `config/train_agxoy_fm.yaml`: Flow Matching defaults (surface eval on, MACE="medium")

### Public API

`dss/__init__.py` exports three functions for use outside of training:
- `get_dataset_agxoy()`: load and return a schnetpack AtomsDataModule
- `get_diffusion_model()`: instantiate ESSFlow + neighbor list from config
- `sample()`: generate structures given a checkpoint, template, and composition list
