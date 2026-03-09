"""Surface evaluation callback: precompute train energies, sample at validation, log metrics and plots.

Uses only dss.helpers, dss.tools.surface_eval, and dss.data.constants (no snowyflow).
"""

import io
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from dss.data.constants.agxoy import (AGXOY_BULK_ENERGIES, AGXOY_REF_ELEMENT,
                                      AGXOY_REF_FORMULA, AGXOY_STOICS)
from dss.helpers import get_energies_for_atoms, sample
from dss.tools import surface_eval as surf

logger = logging.getLogger(__name__)


def _save_val_trajectories(
    trainer: pl.Trainer,
    atoms_trajs: list[list],
    log_dir: Path,
    subdir: str,
    step_idx: int = 0,
    start_idx: int = 0,
) -> None:
    """Write validation sampling trajectories to XYZ files (one multi-frame file per sample)."""
    import ase.io

    traj_dir = log_dir / subdir / f"epoch_{trainer.current_epoch}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    for j, traj in enumerate(atoms_trajs):
        path = traj_dir / f"step_{step_idx:04d}_sample_{start_idx + j:04d}.xyz"
        ase.io.write(path, traj, format="xyz")
    return None


def _infer_sampling_symbols(
    template_atoms,
    num_template: int,
    train_symbol_lists: list[list[str]],
) -> list[str]:
    """Infer mobile species list from training structures (tail after template atoms)."""
    # If template already includes mobile atoms, keep existing behavior.
    template_syms = template_atoms.get_chemical_symbols()[num_template:]
    if template_syms:
        return list(template_syms)
    # Otherwise infer from data: use the most common non-empty tail composition.
    tails = []
    for syms in train_symbol_lists:
        if len(syms) > num_template:
            tail = tuple(syms[num_template:])
            if len(tail) > 0:
                tails.append(tail)
    if tails:
        most_common_tail, _ = Counter(tails).most_common(1)[0]
        return list(most_common_tail)
    return []


def _sample_symbol_tails_from_train(
    train_symbol_lists: list[list[str]],
    num_template: int,
    num_samples: int,
    rng: np.random.Generator,
) -> list[list[str]]:
    """Sample per-structure adsorbate tails from train-set composition distribution."""
    tails = []
    for syms in train_symbol_lists:
        if len(syms) > num_template:
            tail = list(syms[num_template:])
            if len(tail) > 0:
                tails.append(tail)
    if not tails:
        return []
    idx = rng.integers(0, len(tails), size=num_samples)
    return [tails[int(i)] for i in idx]


def _batch_to_symbol_lists(batch: dict, num_template: int) -> list[list[str]]:
    """Extract per-structure symbol lists from a schnetpack-style batch."""
    from ase.data import chemical_symbols

    z = batch.get("_atomic_numbers")
    if z is None:
        return []
    z = z.cpu().numpy()
    n_atoms = batch.get("_n_atoms")
    if n_atoms is None:
        return []
    n_atoms = n_atoms.cpu().numpy()
    out = []
    idx = 0
    for n in n_atoms:
        n = int(n)
        syms = [chemical_symbols[int(z[idx + i])] for i in range(n)]
        out.append(syms)
        idx += n
    return out


def _batch_to_atoms_list(batch: dict) -> list:
    """Convert a schnetpack-style batch to a list of ASE Atoms (same order as _batch_to_symbol_lists)."""
    import ase

    try:
        from schnetpack import \
            properties as prop  # type: ignore[import-untyped]
    except ImportError:
        prop = None
    z = batch.get("_atomic_numbers")
    if z is None and prop is not None:
        z = batch.get(prop.Z)
    pos = batch.get("_positions")
    if pos is None and prop is not None:
        pos = batch.get(prop.R)
    n_atoms = batch.get("_n_atoms")
    if z is None or pos is None or n_atoms is None:
        return []
    z = z.cpu().numpy()
    pos = pos.cpu().numpy()
    n_atoms = n_atoms.cpu().numpy()
    cell = batch.get("_cell")
    if cell is None and prop is not None:
        cell = batch.get(prop.cell)
    if cell is not None:
        cell = cell.cpu().numpy()
    out = []
    idx = 0
    for i, n in enumerate(n_atoms):
        n = int(n)
        numbers = z[idx : idx + n]
        positions = pos[idx : idx + n]
        if cell is not None and cell.size > 0:
            if cell.ndim == 3:
                c = cell[i]
            else:
                c = cell[i * 3 : (i + 1) * 3] if cell.shape[0] >= (i + 1) * 3 else None
            if c is not None and c.size > 0:
                a = ase.Atoms(numbers=numbers, positions=positions, cell=c, pbc=True)
            else:
                a = ase.Atoms(numbers=numbers, positions=positions)
        else:
            a = ase.Atoms(numbers=numbers, positions=positions)
        out.append(a)
        idx += n
    return out


class SurfaceEvalCallback(pl.Callback):
    """Precomputes train energies on fit start; at validation end, samples structures and logs surface metrics."""

    def __init__(
        self,
        template_atoms,
        z_confinement,
        species_names: tuple[str, ...] = ("Ag", "O"),
        num_template: Optional[int] = None,
        val_batch_size: int = 256,
        val_sample_num_steps: int = 100,
        val_sample_postrelax_steps: int = 100,
        val_use_regressor_guidance: bool = False,
        val_guidance_eta: float = 1e-2,
        val_save_trajectories: bool = False,
        val_trajectories_dir: str = "val_trajectories",
        val_surface_chem_pots: Optional[list[dict]] = None,
        target_element: str = "O",
        ref_element: str = "Ag",
        data_type: str = "agxoy",
        mace_model: Optional[str] = None,
        mace_head: str = "omat_pbe",
        mace_device: str = "cuda",
        mace_dtype: str = "float64",
        mace_dispersion: bool = True,
        mace_enable_cueq: bool = False,
        mace_energy_model_instance: Optional[Any] = None,
        train_energies_path: Optional[str] = None,
        val_energy_range: Optional[tuple[float, float]] = None,
        energy_batch_size: int = 32,
    ):
        self.template_atoms = template_atoms
        self.z_confinement = np.asarray(z_confinement, dtype=np.float32)
        self.species_names = species_names
        self.num_template = num_template if num_template is not None else len(template_atoms)
        self.val_batch_size = val_batch_size
        self.val_sample_num_steps = val_sample_num_steps
        self.val_sample_postrelax_steps = val_sample_postrelax_steps
        self.val_use_regressor_guidance = val_use_regressor_guidance
        self.val_guidance_eta = val_guidance_eta
        self.val_save_trajectories = val_save_trajectories
        self.val_trajectories_dir = val_trajectories_dir
        self.val_surface_chem_pots = val_surface_chem_pots or []
        self.target_element = target_element
        self.ref_element = ref_element
        self.data_type = data_type.lower()
        self._mace_config: Optional[dict] = None
        if mace_model is not None:
            self._mace_config = {
                "model": mace_model,
                "head": mace_head,
                "device": mace_device,
                "default_dtype": mace_dtype,
                "dispersion": mace_dispersion,
                "enable_cueq": mace_enable_cueq,
            }
        # Prefer injected MACE instance; fallback lazy-load from _mace_config in on_fit_start.
        self._mace_model = mace_energy_model_instance
        self.val_energy_range = val_energy_range
        self.energy_batch_size = energy_batch_size
        self._train_energies_path_override = (
            Path(train_energies_path).expanduser() if train_energies_path else None
        )

        self._train_energies_path: Optional[Path] = None
        self._train_symbol_lists: Optional[list] = None
        self._val_step_idx: int = 0
        self._val_sampled_energies: list[torch.Tensor] = []
        self._val_sampled_atoms: list = []
        self._val_sampled_symbol_lists: list[list[str]] = []
        self._val_losses: list[float] = []

    def _log_dir(self, trainer: pl.Trainer) -> Path:
        # Keep callback artifacts under the trainer root so users can control
        # a single output directory (e.g., with --wandb_dir).
        d = trainer.default_root_dir
        if d is None:
            d = trainer.log_dir
        if d is None:
            d = "."
        return Path(d)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Precompute train energies and symbol lists; lazily load MACE only as fallback."""
        # Expose callback to LightningModule so validation flow can be orchestrated
        # from VPDiffusion.on_validation_epoch_end.
        setattr(pl_module, "_surface_eval_callback", self)
        log_dir = self._log_dir(trainer)
        self._train_energies_path = self._train_energies_path_override or (log_dir / "train_energies.pt")

        train_energies = None
        if self._train_energies_path.exists():
            train_energies, metadata = surf.load_mace_energies(self._train_energies_path)
            self._train_symbol_lists = metadata.get("symbol_lists", []) if metadata else []
            self.num_template = metadata.get("num_template", self.num_template) if metadata else self.num_template
            
            # If symbol_lists missing from metadata, we need to re-scan training data
            if not self._train_symbol_lists:
                logger.info("symbol_lists missing from %s. Re-scanning training data...", self._train_energies_path)
                dataloader = trainer.datamodule.train_dataloader()
                self._train_symbol_lists = []
                for batch in dataloader:
                    self._train_symbol_lists.extend(_batch_to_symbol_lists(batch, self.num_template))
                # Update the file with metadata if possible
                metadata = metadata or {}
                metadata["symbol_lists"] = self._train_symbol_lists
                metadata["num_template"] = self.num_template
                surf.save_mace_energies(train_energies, self._train_energies_path, metadata=metadata)
            return

        use_mace = self._mace_model is not None or self._mace_config is not None
        use_potential = getattr(pl_module, "potential_model", None) is not None
        if not use_mace and not use_potential:
            return

        train_symbol_lists = []
        dataloader = trainer.datamodule.train_dataloader()

        if use_mace:
            if self._mace_model is None:
                from dss.data.constants.agxoy import mask_index
                from dss.data.constants.agxoy import \
                    number_to_element as agxoy_number_to_element
                number_to_element = {k: v for k, v in agxoy_number_to_element.items() if k <= mask_index}
                from dss.energy.mace import MACEEnergyModel

                self._mace_model = MACEEnergyModel(
                    number_to_element=number_to_element,
                    **self._mace_config,
                )
            train_atoms = []
            for batch in dataloader:
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                train_atoms.extend(_batch_to_atoms_list(batch))
                train_symbol_lists.extend(_batch_to_symbol_lists(batch, self.num_template))
            train_energies = surf.precompute_mace_energies(
                train_atoms, self._mace_model, batch_size=self.energy_batch_size
            )

            metadata = {
                "model": self._mace_model._model_path,
                "head": self._mace_model._head,
                "dispersion": self._mace_model._dispersion,
                "n_samples": len(train_energies),
                "symbol_lists": train_symbol_lists,
                "num_template": self.num_template,
            }
        else:
            train_energies_list = []
            pl_module.eval()
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    out = pl_module.potential(batch)
                    e = out["energy"]
                    if e.dim() == 0:
                        e = e.unsqueeze(0)
                    elif e.size(0) != len(batch["_idx"]):
                        e = e.view(len(batch["_idx"]), -1).sum(1)
                    train_energies_list.append(e.cpu())
                    train_symbol_lists.extend(_batch_to_symbol_lists(batch, self.num_template))
            pl_module.train()
            train_energies = torch.cat(train_energies_list, dim=0)

            metadata = {
                "n_samples": len(train_energies),
                "symbol_lists": train_symbol_lists,
                "num_template": self.num_template,
            }

        self._train_symbol_lists = train_symbol_lists
        surf.save_mace_energies(train_energies, self._train_energies_path, metadata=metadata)


        for key, val in [
            ("precompute/train_energy_mean", train_energies.mean().item()),
            ("precompute/train_energy_std", train_energies.std().item()),
            ("precompute/train_energy_min", train_energies.min().item()),
            ("precompute/train_energy_max", train_energies.max().item()),
            ("precompute/train_n_structures", len(train_energies)),
        ]:
            trainer.logger.log_metrics({key: val}, step=trainer.global_step)

    def _load_train_reference(self, trainer: pl.Trainer) -> tuple[torch.Tensor, list[list[str]], int] | None:
        """Load train energies/symbols reference used for composition-energy comparison."""
        log_dir = self._log_dir(trainer)
        if self._train_energies_path is None:
            self._train_energies_path = self._train_energies_path_override or (log_dir / "train_energies.pt")
        if not self._train_energies_path.exists():
            return None
        train_energies, metadata = surf.load_mace_energies(self._train_energies_path)
        self._train_symbol_lists = metadata.get("symbol_lists", []) if metadata else []
        num_template = metadata.get("num_template", self.num_template) if metadata else self.num_template
        return train_energies, self._train_symbol_lists, num_template

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset per-epoch accumulators for sampled validation outputs."""
        self._val_step_idx = 0
        self._val_sampled_energies = []
        self._val_sampled_atoms = []
        self._val_sampled_symbol_lists = []
        self._val_losses = []

    def run_validation_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Optional[dict] = None,
        val_loss: Optional[float] = None,
    ) -> None:
        """Run one sampling/eval pass during a validation step and collect outputs."""
        if getattr(pl_module, "potential_model", None) is None:
            return
        ref = self._load_train_reference(trainer)
        if ref is None:
            return
        _, train_symbol_lists, num_template = ref
        val_symbol_lists = _batch_to_symbol_lists(batch, num_template) if batch is not None else []
        log_dir = self._log_dir(trainer)


        # Sample on each validation step (snowyflow-style accumulation)
        rng = np.random.default_rng(trainer.current_epoch * 100000 + self._val_step_idx)
        symbol_tails = _sample_symbol_tails_from_train(
            val_symbol_lists or train_symbol_lists or [],
            num_template,
            self.val_batch_size,
            rng,
        )

        if symbol_tails:
            symbols = symbol_tails  # per-sample compositions drawn from val distribution (fallback: train)
        else:
            symbols = _infer_sampling_symbols(
                self.template_atoms, num_template, val_symbol_lists or self._train_symbol_lists or []
            )
            if not symbols:
                # Keep old fallback, but this is likely a slab-only setup.
                symbols = list(self.species_names) * (len(self.template_atoms) - num_template)
                if trainer.logger is not None:
                    trainer.logger.log_metrics(
                        {"val/warn_no_mobile_symbols": 1.0}, step=trainer.global_step
                    )

        out = sample(
            pl_module,
            self.val_batch_size,
            self.template_atoms,
            symbols,
            self.z_confinement,
            num_steps=self.val_sample_num_steps,
            eta=self.val_guidance_eta,
            postrelax_steps=self.val_sample_postrelax_steps,
            return_trajectories=self.val_save_trajectories,
            use_regressor_guidance=self.val_use_regressor_guidance,
        )
        if self.val_save_trajectories:
            sampled_atoms, atoms_trajs = out
            _save_val_trajectories(
                trainer,
                atoms_trajs,
                log_dir,
                self.val_trajectories_dir,
                step_idx=self._val_step_idx,
                start_idx=self._val_step_idx * self.val_batch_size,
            )
        else:
            sampled_atoms = out

        # Energies for sampled (potential or MACE)
        if self._mace_model is not None:
            sampled_energies = surf.precompute_mace_energies(
                sampled_atoms, self._mace_model, batch_size=self.energy_batch_size
            )
        else:
            sampled_energies = get_energies_for_atoms(
                pl_module, sampled_atoms, num_template, self.z_confinement, batch_size=self.energy_batch_size
            )
        self._val_sampled_energies.append(sampled_energies.detach().cpu())
        self._val_sampled_atoms.extend(sampled_atoms)
        self._val_sampled_symbol_lists.extend([a.get_chemical_symbols() for a in sampled_atoms])
        if val_loss is not None:
            self._val_losses.append(float(val_loss))
        self._val_step_idx += 1

    def finalize_validation_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Aggregate stepwise sampled outputs and log/plot epoch-level metrics."""
        ref = self._load_train_reference(trainer)
        if ref is None:
            return

        train_energies, train_symbol_lists, num_template = ref
        if len(self._val_sampled_energies) == 0:
            return
        sampled_energies = torch.cat(self._val_sampled_energies, dim=0)
        sampled_symbol_lists = self._val_sampled_symbol_lists
        sampled_atoms = self._val_sampled_atoms

        # Snowyflow-style: optionally filter sampled structures by validation energy range.
        if self.val_energy_range is not None and len(self.val_energy_range) == 2:
            min_e, max_e = float(self.val_energy_range[0]), float(self.val_energy_range[1])
            n_before = int(sampled_energies.numel())
            mask = (sampled_energies >= min_e) & (sampled_energies <= max_e)
            num_dropped = int((~mask).sum().item())
            if num_dropped > 0:
                keep_idx = torch.nonzero(mask, as_tuple=False).flatten().tolist()
                sampled_energies = sampled_energies[mask]
                sampled_symbol_lists = [sampled_symbol_lists[i] for i in keep_idx]
                sampled_atoms = [sampled_atoms[i] for i in keep_idx]
                if trainer.logger is not None:
                    trainer.logger.log_metrics(
                        {"val/fraction_out_of_energy_range": float(num_dropped) / float(max(n_before, 1))},
                        step=trainer.global_step,
                    )
            if sampled_energies.numel() == 0:
                if trainer.logger is not None:
                    trainer.logger.log_metrics(
                        {"val/warn_all_filtered_by_energy_range": 1.0},
                        step=trainer.global_step,
                    )
                return

        # Compositions
        train_compositions = surf.compositions_from_symbol_lists(
            train_symbol_lists, [num_template] * len(train_symbol_lists), self.species_names
        )
        sampled_compositions = surf.compositions_from_symbol_lists(
            sampled_symbol_lists, [num_template] * len(sampled_symbol_lists), self.species_names
        )

        energies_dict = {"Train": train_energies, "dss": sampled_energies}
        compositions_dict = {"Train": train_compositions, "dss": sampled_compositions}

        # Scalar metrics
        metrics = surf.energy_comparison_metrics(energies_dict, compositions_dict)
        for k, v in metrics.items():
            trainer.logger.log_metrics({f"val/{k}": v}, step=trainer.global_step)
        comp_w = surf.wasserstein_composition(train_compositions, sampled_compositions, use_first_species_only=True)
        trainer.logger.log_metrics({"val/composition_wasserstein": comp_w}, step=trainer.global_step)
        if len(self._val_losses) > 0:
            trainer.logger.log_metrics({"val/loss_sample_mean": float(np.mean(self._val_losses))}, step=trainer.global_step)

        # Energy distribution figure
        fig = surf.plot_energy_distribution(
            {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in energies_dict.items()},
            title="Energy distribution (Train vs Sampled)",
        )
        self._log_figure(trainer, fig, "val/energy_distribution")
        plt.close(fig)

        # Energy per composition figure
        fig = surf.plot_energy_per_composition(
            energies_dict, compositions_dict, title="Energy by composition (Train vs Sampled)"
        )
        self._log_figure(trainer, fig, "val/energy_per_composition")
        plt.close(fig)

        # Surface stability plots (if chem pots given)
        if self.val_surface_chem_pots and self.data_type == "agxoy":
            offset_data = {
                "bulk_energies": dict(AGXOY_BULK_ENERGIES),
                "stoics": AGXOY_STOICS,
                "ref_formula": AGXOY_REF_FORMULA,
                "ref_element": AGXOY_REF_ELEMENT,
            }
            if len(sampled_atoms) == len(sampled_energies):
                data_list = [
                    {"atoms": a, "energy": float(sampled_energies[i]), "label": "VSSR-MC sample"}
                    for i, a in enumerate(sampled_atoms)
                ]
                energies_list = [float(sampled_energies[i]) for i in range(len(sampled_atoms))]
                try:
                    figs = surf.plot_surface_stability(
                        data_list,
                        self.val_surface_chem_pots,
                        save_dir=None,
                        energies=energies_list,
                        offset_data=offset_data,
                        target_element=self.target_element,
                        ref_element=self.ref_element,
                    )
                    for i, fig in enumerate(figs):
                        self._log_figure(trainer, fig, f"val/surface_stability_{i}")
                        plt.close(fig)
                except Exception:
                    # Keep other validation plots/metrics even if surface stability fails.
                    if trainer.logger is not None:
                        trainer.logger.log_metrics({"val/warn_surface_stability_failed": 1.0}, step=trainer.global_step)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Validation is orchestrated in VPDiffusion hooks."""
        return

    def _log_figure(self, trainer: pl.Trainer, fig, key: str) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        if hasattr(trainer.logger, "log_image"):
            try:
                import wandb
                trainer.logger.log_image(
                    key=key,
                    images=[wandb.Image(fig)],
                    step=trainer.global_step,
                )
                return
            except Exception:
                pass
        if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "add_figure"):
            trainer.logger.experiment.add_figure(key, fig, trainer.global_step)
            return
        save_path = self._log_dir(trainer) / f"{key.replace('/', '_')}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        buf.close()
