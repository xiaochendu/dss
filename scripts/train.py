#!/usr/bin/env python3
"""Train DSS diffusion model on AgxOy surface data (same dataset layout as snowyflow).

Examples:
    python scripts/train.py --data_path /path/to/agox_AgxOy_structures --max_epochs 2

    # From config (override from CLI):
    python scripts/train.py --config config/train_agxoy.yaml --data_path /path/to/data

    # With GNN size, RBFs, and Wandb:
    python scripts/train.py --data_path /path/to/data --n_atom_basis 128 --n_rbf 50 \\
        --wandb --wandb_project my_project --wandb_run agxoy_v1 --max_epochs 10

    # With surface eval (sample at validation, log energy/composition metrics and plots):
    python scripts/train.py --data_path /path/to/data --surface_eval --val_sample_num_samples 256

Wandb: Use --wandb to log to Weights & Biases. Set WANDB_ENTITY or log in with
    wandb login. Logs are written to --wandb_dir (default: <data_path>/dss_run/wandb).

Surface eval: Use --surface_eval to enable validation-time sampling and surface metrics
    (energy distribution, composition Wasserstein, energy per composition, optional surface stability).
    With --mace_model, the callback loads the in-dss MACEEnergyModel in on_fit_start and uses MACE
    for train and sampled energies. Requires the mace package.
"""
import argparse
import sys
from pathlib import Path

# Ensure dss is importable when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as pl
import yaml

from dss import get_dataset_agxoy, get_diffusion_model
from dss.callbacks import SurfaceEvalCallback
from dss.utils import TorchNeighborList


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _format_wandb_template(s: str, args) -> str:
    """If s contains '{...}' placeholders, format with args (e.g. {n_atom_basis}, {batch_size})."""
    if not s or "{" not in s:
        return s
    try:
        return s.format(**vars(args))
    except KeyError:
        return s


def main():
    p = argparse.ArgumentParser(description="Train DSS on AgxOy surface structures")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config (e.g. config/train_agxoy.yaml); CLI overrides config")
    p.add_argument("--data_path", type=str, default=None, help="Directory with template_*.xyz and agox_sample_AgxOy_*.xyz")
    p.add_argument("--mcmc_files", type=str, nargs="*", default=None, help="MCMC XYZ filenames only (e.g. agox_sample_AgxOy_2000.xyz). If not set, glob agox_sample_AgxOy_*.xyz")
    p.add_argument("--path", type=str, default="dataset.db", help="Path for created .db and split.npz")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--max_epochs", type=int, default=100, help="Max training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--cutoff", type=float, default=6.0, help="Cutoff for neighbour list (Ang)")
    p.add_argument("--n_atom_basis", type=int, default=64, help="Number of GNN (PaiNN) atom features")
    p.add_argument("--n_rbf", type=int, default=30, help="Number of radial basis functions")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (0 for stability)")
    p.add_argument("--limit_train_batches", type=int, default=None, help="If set, limit train batches per epoch (for quick test)")
    p.add_argument("--limit_val_batches", type=int, default=None, help="If set, limit val batches per epoch")
    p.add_argument("--val_check_interval", type=float, default=1.0, help="Run validation every N fraction of epoch (1.0=every epoch, 0.5=twice per epoch, 2.0=every 2 epochs). Passed to Trainer.")
    # Wandb
    p.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging")
    p.add_argument("--wandb_project", type=str, default="dss", help="Wandb project name")
    p.add_argument("--wandb_run", type=str, default=None, help="Wandb run name (default: auto)")
    p.add_argument("--wandb_dir", type=str, default=None, help="Wandb save_dir (default: <run_dir>/wandb or ./wandb)")
    # Surface eval (validation-time sampling + metrics)
    p.add_argument("--surface_eval", action="store_true", help="Enable surface eval callback (sample at val, log energy/composition metrics and plots)")
    p.add_argument("--val_sample_num_samples", type=int, default=256, help="Number of structures to sample per validation for surface eval")
    p.add_argument("--val_sample_num_steps", type=int, default=100, help="Diffusion time steps per validation sampling (default 100)")
    p.add_argument("--val_sample_postrelax_steps", type=int, default=0, help="Postrelaxation time steps per validation sampling (default 0)")
    p.add_argument("--val_save_trajectories", action="store_true", help="Save validation sampling trajectories as XYZ (one multi-frame file per sample, under val_trajectories_dir)")
    p.add_argument("--val_trajectories_dir", type=str, default="val_trajectories", help="Subdir under log dir for trajectory XYZ files (default: val_trajectories)")
    # MACE (optional; used by surface eval callback when set)
    p.add_argument("--mace_model", type=str, default=None, help="MACE model path or URL; if set, callback loads MACE in on_fit_start for train/sampled energies")
    p.add_argument("--mace_head", type=str, default="omat_pbe", help="MACE head (default: omat_pbe)")
    p.add_argument("--mace_device", type=str, default="cuda", help="MACE device (default: cuda)")
    p.add_argument("--mace_dtype", type=str, default="float64", help="MACE default_dtype (default: float64)")
    p.add_argument("--mace_dispersion", action="store_true", default=True, help="MACE D3 dispersion (default: True)")
    p.add_argument("--no_mace_dispersion", action="store_false", dest="mace_dispersion", help="Disable MACE dispersion")
    p.add_argument("--mace_enable_cueq", action="store_true", default=False, help="MACE cuEquivariance (default: False)")
    # Load config first so CLI overrides config file
    args_pre, _ = p.parse_known_args()
    if args_pre.config is not None:
        cfg = _load_config(args_pre.config)
        for action in p._actions:
            if action.dest in cfg and action.dest != "config":
                action.default = cfg[action.dest]
        if cfg.get("data_path") is not None:
            # so we don't require data_path on CLI when set in config
            for action in p._actions:
                if action.dest == "data_path":
                    action.required = False
                    break
    args = p.parse_args()
    if args.data_path is None:
        p.error("--data_path is required (or set data_path in --config YAML)")

    # Shared neighbour list for dataset and model (same cutoff)
    neighbour_list = TorchNeighborList(args.cutoff)

    # Use absolute path for db/split so all processes see the same file (avoid readonly errors)
    if Path(args.path).is_absolute():
        db_path = args.path
        split_path = str(Path(args.path).parent / "split.npz")
    else:
        run_dir = Path(args.data_path).resolve() / "dss_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(run_dir / (args.path if args.path != "dataset.db" else "dataset.db"))
        split_path = str(run_dir / "split.npz")

    # Load AgxOy dataset (same layout as snowyflow)
    datamodule, template_atoms, z_confinement = get_dataset_agxoy(
        args.data_path,
        mcmc_xyz_files=args.mcmc_files,
        path=db_path,
        split_file=split_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cutoff=args.cutoff,
        neighbour_list=neighbour_list,
    )

    # Build diffusion model
    diffusion, _ = get_diffusion_model(
        cutoff=args.cutoff,
        n_atom_basis=args.n_atom_basis,
        n_rbf=args.n_rbf,
        lr=args.lr,
        neighbour_list=neighbour_list,
    )

    # Logger: Wandb if requested, else default TensorBoard
    logger = True
    if args.wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError:
            raise ImportError("Wandb logging requires: pip install wandb")
        wandb_save_dir = args.wandb_dir
        if wandb_save_dir is None:
            run_dir = Path(args.data_path).resolve() / "dss_run"
            wandb_save_dir = str(run_dir / "wandb")
        logger = WandbLogger(
            project=_format_wandb_template(args.wandb_project, args),
            name=_format_wandb_template(args.wandb_run, args) or None,
            save_dir=wandb_save_dir,
        )

    # Callbacks
    callbacks = []
    if args.surface_eval:
        mace_kwargs = {}
        if args.mace_model is not None:
            mace_kwargs = {
                "mace_model": args.mace_model,
                "mace_head": args.mace_head,
                "mace_device": args.mace_device,
                "mace_dtype": args.mace_dtype,
                "mace_dispersion": args.mace_dispersion,
                "mace_enable_cueq": args.mace_enable_cueq,
            }
        callbacks.append(
            SurfaceEvalCallback(
                template_atoms=template_atoms,
                z_confinement=z_confinement,
                species_names=("Ag", "O"),
                num_template=len(template_atoms),
                val_sample_num_samples=args.val_sample_num_samples,
                val_sample_num_steps=args.val_sample_num_steps,
                val_sample_postrelax_steps=args.val_sample_postrelax_steps,
                val_save_trajectories=args.val_save_trajectories,
                val_trajectories_dir=args.val_trajectories_dir,
                **mace_kwargs,
            )
        )

    # Train (single device to avoid multi-process db issues)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
    )
    trainer.fit(diffusion, datamodule)
    print("Training finished. Template and z_confinement available for sampling.")


if __name__ == "__main__":
    main()
