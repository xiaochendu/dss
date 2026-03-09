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
    python scripts/train.py --data_path /path/to/data --surface_eval --val_batch_size 256

Wandb: Use --wandb to log to Weights & Biases. Set WANDB_ENTITY or log in with
    wandb login. By default W&B files are written to the run directory
    (<run_root>/<run_dir_template>), unless --wandb_dir is explicitly set.

Surface eval: Use --surface_eval to enable validation-time sampling and surface metrics
    (energy distribution, composition Wasserstein, energy per composition, optional surface stability).
    With --mace_model, the callback loads the in-dss MACEEnergyModel in on_fit_start and uses MACE
    for train and sampled energies. Requires the mace package.
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

# Ensure dss is importable when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as pl
import yaml

from dss import get_dataset_agxoy, get_diffusion_model
from dss.callbacks import SurfaceEvalCallback
from dss.data.constants.agxoy import mask_index
from dss.data.constants.agxoy import \
    number_to_element as agxoy_number_to_element
from dss.energy.mace import MACEEnergyModel
from dss.utils import TorchNeighborList


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _format_wandb_template(s: str, args) -> str:
    """If s contains '{...}' placeholders, format with args (e.g. {n_atom_basis}, {batch_size})."""
    if not s or "{" not in s:
        return s
    # Support both "{var}" and "${var}" styles in config templates.
    s = re.sub(r"\$\{([^}:]+)\}", r"{\1}", s)
    try:
        return s.format(**vars(args))
    except KeyError:
        return s


def _format_run_dir_template(template: str, args) -> str:
    """Format run dir template with args and {now:...} timestamp token."""
    if not template:
        return ""
    out = template
    if "{now:" in out:
        i = out.find("{now:")
        j = out.find("}", i)
        if j != -1:
            dt_fmt = out[i + 5 : j]
            out = out[:i] + datetime.now().strftime(dt_fmt) + out[j + 1 :]
    out = _format_wandb_template(out, args)
    return out


def main():
    p = argparse.ArgumentParser(description="Train DSS on AgxOy surface structures")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config (e.g. config/train_agxoy.yaml); CLI overrides config")
    p.add_argument("--data_path", type=str, default=None, help="Directory with template_*.xyz and agox_sample_AgxOy_*.xyz")
    p.add_argument("--val_data_path", type=str, default=None, help="Optional separate directory for validation dataset. If set, val split is built from this path.")
    p.add_argument("--mcmc_files", type=str, nargs="*", default=None, help="MCMC XYZ filenames only (e.g. agox_sample_AgxOy_2000.xyz). If not set, glob agox_sample_AgxOy_*.xyz")
    p.add_argument("--path", type=str, default="dataset.db", help="Path for created .db and split.npz")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--num_train", type=float, default=0.9, help="Fraction of data for training")
    p.add_argument("--num_val", type=float, default=0.1, help="Fraction for validation")
    p.add_argument("--reuse_train_for_val", action="store_true", help="Reuse training split as validation dataset (useful with num_train=1.0 while keeping validation hooks/callbacks active).")
    p.add_argument("--max_epochs", type=int, default=100, help="Max training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--cutoff", type=float, default=6.0, help="Cutoff for neighbour list (Ang)")
    p.add_argument("--n_atom_basis", type=int, default=64, help="Number of GNN (PaiNN) atom features")
    p.add_argument("--n_rbf", type=int, default=30, help="Number of radial basis functions")
    p.add_argument("--n_interactions", type=int, default=4, help="Number of GNN interaction layers")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers (0 for stability)")
    p.add_argument("--limit_train_batches", type=int, default=None, help="If set, limit train batches per epoch (for quick test)")
    p.add_argument("--limit_val_batches", type=int, default=None, help="If set, limit val batches per epoch")
    p.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Run validation every N epochs (default: 1).")
    p.add_argument("--val_check_interval", type=float, default=None, help="Optional intra-epoch validation interval passed to Trainer (e.g. 0.5=twice/epoch). If unset, epoch-based scheduling is used.")
    # Checkpointing (snowyflow-style)
    p.add_argument("--checkpoint_monitor", type=str, default="val/energy_wasserstein_pooled", help="Metric to monitor for ModelCheckpoint")
    p.add_argument("--checkpoint_mode", type=str, default="min", choices=["min", "max"], help="ModelCheckpoint mode")
    p.add_argument("--checkpoint_save_top_k", type=int, default=1, help="Save top-k checkpoints by monitored metric")
    p.add_argument("--checkpoint_save_last", action="store_true", default=True, help="Also save last checkpoint")
    p.add_argument("--no_checkpoint_save_last", action="store_false", dest="checkpoint_save_last", help="Disable saving last checkpoint")
    # Wandb
    p.add_argument("--wandb", action="store_true", help="Enable online Weights & Biases sync. If omitted, wandb runs in offline mode.")
    p.add_argument("--wandb_project", type=str, default="dss", help="Wandb project name")
    p.add_argument("--wandb_run", type=str, default=None, help="Wandb run name (default: auto)")
    p.add_argument("--wandb_id", type=str, default=None, help="Wandb run ID for resuming (default: None)")
    p.add_argument("--wandb_dir", type=str, default=None, help="Wandb save_dir override. If null, uses run dir.")
    # Run/output directory (hydra-like)
    p.add_argument("--run_root", type=str, default="./outputs/agxoy", help="Base output directory for this experiment family")
    p.add_argument("--run_dir_template", type=str, default="dss_nf{n_atom_basis}_nr{n_rbf}_bs{batch_size}_lr{lr}/{now:%Y-%m-%d_%H-%M-%S}", help="Run subdir template under run_root. Supports arg placeholders and {now:strftime}.")
    # Surface eval (validation-time sampling + metrics)
    p.add_argument("--surface_eval", action="store_true", help="Enable surface eval callback (sample at val, log energy/composition metrics and plots)")
    p.add_argument("--val_batch_size", type=int, default=256, help="Number of structures to sample per validation for surface eval")
    p.add_argument("--val_sample_num_steps", type=int, default=100, help="Diffusion time steps per validation sampling (default 100)")
    p.add_argument("--val_sample_postrelax_steps", type=int, default=0, help="Postrelaxation time steps per validation sampling (default 0)")
    p.add_argument("--val_use_regressor_guidance", action="store_true", help="Use regressor_guidance_sample for validation sampling. Default uses VPDiffusion.sample (unguided).")
    p.add_argument("--val_guidance_eta", type=float, default=1e-2, help="Guidance strength eta for regressor_guidance_sample when enabled.")
    p.add_argument("--val_energy_range", type=yaml.safe_load, default=None, help="Optional sampled-energy filter range for val metrics/plots, e.g. \"[-100,200]\"")
    p.add_argument("--val_surface_chem_pots", type=yaml.safe_load, default=[], help="Surface stability chemical potentials, e.g. \"[{Ag:0.0,O:0.0},{Ag:-1.0,O:0.0}]\"")
    p.add_argument("--train_energies_path", type=str, default=None, help="Optional path to precomputed train_energies.pt for surface eval (if missing, it will be computed and saved there)")
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
    p.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for resuming training (full state)")
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

    # Build run root (hydra-like run dir template)
    run_subdir = _format_run_dir_template(args.run_dir_template, args)
    run_root = (Path(args.run_root).expanduser() / run_subdir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    # Shared neighbour list for dataset and model (same cutoff)
    neighbour_list = TorchNeighborList(args.cutoff)

    # Use absolute path for db/split so all processes see the same file (avoid readonly errors)
    if Path(args.path).is_absolute():
        db_path = args.path
        split_path = str(Path(args.path).parent / "split.npz")
    else:
        db_path = str(run_root / (args.path if args.path != "dataset.db" else "dataset.db"))
        split_path = str(run_root / "split.npz")

    # Load AgxOy dataset (same layout as snowyflow)
    datamodule, template_atoms, z_confinement = get_dataset_agxoy(
        args.data_path,
        mcmc_xyz_files=args.mcmc_files,
        path=db_path,
        split_file=split_path,
        batch_size=args.batch_size,
        num_train=args.num_train,
        num_val=args.num_val,
        num_workers=args.num_workers,
        cutoff=args.cutoff,
        neighbour_list=neighbour_list,
    )

    # Optional validation dataset override.
    # - If val_data_path is set: build a val-only split from that path and use it as datamodule val set.
    # - Else if requested: reuse train split as val (keeps validation hooks active with num_val=0).
    if args.val_data_path is not None:
        val_db_path = str(run_root / "val_dataset.db")
        val_split_path = str(run_root / "val_split.npz")
        val_dm, _, _ = get_dataset_agxoy(
            args.val_data_path,
            mcmc_xyz_files=args.mcmc_files,
            path=val_db_path,
            split_file=val_split_path,
            batch_size=args.val_batch_size,
            num_train=0.0,
            num_val=1.0,
            num_workers=args.num_workers,
            cutoff=args.cutoff,
            neighbour_list=neighbour_list,
        )
        datamodule._val_dataset = val_dm.val_dataset
        datamodule._val_dataloader = None
    elif args.reuse_train_for_val:
        val_dm, _,_ = get_dataset_agxoy(
        args.data_path,
        mcmc_xyz_files=args.mcmc_files,
        path=db_path,
        split_file=split_path,
        batch_size=args.val_batch_size if args.val_batch_size is not None else args.batch_size,
        num_train=0.0,
        num_val=1.0,
        num_workers=args.num_workers,   
        cutoff=args.cutoff,
        neighbour_list=neighbour_list,
    )
        datamodule._val_dataset = val_dm.val_dataset
        datamodule._val_dataloader = None

    # Optional centralized MACE initialization.
    mace_energy_model = None
    if args.mace_model is not None:
        number_to_element = {k: v for k, v in agxoy_number_to_element.items() if k <= mask_index}
        mace_energy_model = MACEEnergyModel(
            model=args.mace_model,
            head=args.mace_head,
            device=args.mace_device,
            default_dtype=args.mace_dtype,
            dispersion=args.mace_dispersion,
            enable_cueq=args.mace_enable_cueq,
            number_to_element=number_to_element,
        )

    # Build diffusion model
    diffusion, _ = get_diffusion_model(
        cutoff=args.cutoff,
        n_atom_basis=args.n_atom_basis,
        n_rbf=args.n_rbf,
        n_interactions=args.n_interactions,
        lr=args.lr,
        neighbour_list=neighbour_list,
        potential_model_instance=mace_energy_model,
    )

    # Logger: always use WandbLogger; --wandb toggles online sync.
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError:
        raise ImportError("Wandb logging requires: pip install wandb")
    wandb_save_dir = str(Path(args.wandb_dir).expanduser().resolve()) if args.wandb_dir else str(run_root)
    logger_kwargs = dict(
        project=_format_wandb_template(args.wandb_project, args),
        name=_format_wandb_template(args.wandb_run, args) or None,
        save_dir=wandb_save_dir,
        offline=not args.wandb,
    )
    if args.wandb_id:
        logger_kwargs["id"] = args.wandb_id
        logger_kwargs["resume"] = "must"
    logger = WandbLogger(**logger_kwargs)

    # Callbacks
    callbacks = []
    from pytorch_lightning.callbacks import ModelCheckpoint

    callbacks.append(
        ModelCheckpoint(
            monitor=args.checkpoint_monitor,
            mode=args.checkpoint_mode,
            save_top_k=args.checkpoint_save_top_k,
            save_last=args.checkpoint_save_last,
            dirpath=str(run_root / "checkpoints"),
            filename="{epoch:02d}-val_wasserstein-{" + args.checkpoint_monitor + ":.4f}",
            auto_insert_metric_name=False,
            verbose=True,
        )
    )

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
                val_batch_size=args.val_batch_size,
                val_sample_num_steps=args.val_sample_num_steps,
                val_sample_postrelax_steps=args.val_sample_postrelax_steps,
                val_use_regressor_guidance=args.val_use_regressor_guidance,
                val_guidance_eta=args.val_guidance_eta,
                val_energy_range=args.val_energy_range,
                val_surface_chem_pots=args.val_surface_chem_pots,
                train_energies_path=args.train_energies_path,
                val_save_trajectories=args.val_save_trajectories,
                val_trajectories_dir=args.val_trajectories_dir,
                mace_energy_model_instance=mace_energy_model,
                **mace_kwargs,
            )
        )

    # Train (single device to avoid multi-process db issues)
    trainer_kwargs = dict(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        default_root_dir=str(run_root),
        callbacks=callbacks,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )
    if args.val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = args.val_check_interval
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(diffusion, datamodule, ckpt_path=args.ckpt_path)
    print("Training finished. Template and z_confinement available for sampling.")


if __name__ == "__main__":
    main()
