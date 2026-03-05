"""Surface evaluation callback: precompute train energies, sample at validation, log metrics and plots.

Uses only dss.helpers, dss.tools.surface_eval, and dss.data.constants (no snowyflow).
"""

import io
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


def _save_val_trajectories(
    trainer: pl.Trainer,
    atoms_trajs: list[list],
    log_dir: Path,
    subdir: str,
) -> None:
    """Write validation sampling trajectories to XYZ files (one multi-frame file per sample)."""
    import ase.io

    traj_dir = log_dir / subdir / f"epoch_{trainer.current_epoch}"
    traj_dir.mkdir(parents=True, exist_ok=True)
    for j, traj in enumerate(atoms_trajs):
        path = traj_dir / f"sample_{j:04d}.xyz"
        ase.io.write(path, traj, format="xyz")
    return None


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
        val_sample_num_samples: int = 256,
        val_sample_num_steps: int = 100,
        val_sample_postrelax_steps: int = 100,
        val_save_trajectories: bool = False,
        val_trajectories_dir: str = "val_trajectories",
        val_surface_chem_pots: Optional[list[dict]] = None,
        target_element: str = "Ag",
        ref_element: str = "O",
        data_type: str = "agxoy",
        mace_model: Optional[str] = None,
        mace_head: str = "omat_pbe",
        mace_device: str = "cuda",
        mace_dtype: str = "float64",
        mace_dispersion: bool = True,
        mace_enable_cueq: bool = False,
        val_energy_range: Optional[tuple[float, float]] = None,
        energy_batch_size: int = 32,
    ):
        self.template_atoms = template_atoms
        self.z_confinement = np.asarray(z_confinement, dtype=np.float32)
        self.species_names = species_names
        self.num_template = num_template if num_template is not None else len(template_atoms)
        self.val_sample_num_samples = val_sample_num_samples
        self.val_sample_num_steps = val_sample_num_steps
        self.val_sample_postrelax_steps = val_sample_postrelax_steps
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
        self._mace_model = None  # loaded in on_fit_start when _mace_config is set
        self.val_energy_range = val_energy_range
        self.energy_batch_size = energy_batch_size

        self._train_energies_path: Optional[Path] = None
        self._train_symbol_lists: Optional[list] = None

    def _log_dir(self, trainer: pl.Trainer) -> Path:
        d = trainer.log_dir
        if d is None:
            d = Path(trainer.default_root_dir or ".")
        return Path(d)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Precompute train energies (and symbol lists) and save to log_dir. Load MACE here when config set."""
        log_dir = self._log_dir(trainer)
        self._train_energies_path = log_dir / "train_energies.pt"
        if self._train_energies_path.exists():
            data = torch.load(self._train_energies_path, weights_only=False)
            self._train_symbol_lists = data.get("symbol_lists", [])
            return

        use_mace = self._mace_config is not None
        use_potential = getattr(pl_module, "potential_model", None) is not None
        if not use_mace and not use_potential:
            return

        train_symbol_lists = []
        dataloader = trainer.datamodule.train_dataloader()

        if use_mace:
            from dss.data.constants.agxoy import mask_index
            from dss.data.constants.agxoy import \
                number_to_element as agxoy_number_to_element
            from dss.energy.mace import MACEEnergyModel

            number_to_element = {k: v for k, v in agxoy_number_to_element.items() if k <= mask_index}
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

        self._train_symbol_lists = train_symbol_lists
        log_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"energies": train_energies, "symbol_lists": train_symbol_lists, "num_template": self.num_template},
            self._train_energies_path,
        )

        for key, val in [
            ("precompute/train_energy_mean", train_energies.mean().item()),
            ("precompute/train_energy_std", train_energies.std().item()),
            ("precompute/train_energy_min", train_energies.min().item()),
            ("precompute/train_energy_max", train_energies.max().item()),
            ("precompute/train_n_structures", len(train_energies)),
        ]:
            trainer.logger.log_metrics({key: val}, step=trainer.global_step)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Sample structures, compute energies, run surface_eval and log metrics/images."""
        if getattr(pl_module, "potential_model", None) is None:
            return
        log_dir = self._log_dir(trainer)

        # Load train energies (and symbol lists)
        if self._train_energies_path is None:
            self._train_energies_path = log_dir / "train_energies.pt"
        if not self._train_energies_path.exists():
            return
        data = torch.load(self._train_energies_path, weights_only=False)
        train_energies = data["energies"]
        self._train_symbol_lists = data.get("symbol_lists", [])
        num_template = data.get("num_template", self.num_template)

        # Sample (optionally with trajectories for inspection)
        symbols = self.template_atoms.get_chemical_symbols()[num_template:]
        if not symbols:
            symbols = list(self.species_names) * (len(self.template_atoms) - num_template)
        out = sample(
            pl_module,
            self.val_sample_num_samples,
            self.template_atoms,
            symbols,
            self.z_confinement,
            num_steps=self.val_sample_num_steps,
            postrelax_steps=self.val_sample_postrelax_steps,
            return_trajectories=self.val_save_trajectories,
        )
        if self.val_save_trajectories:
            sampled_atoms, atoms_trajs = out
            _save_val_trajectories(
                trainer, atoms_trajs, log_dir, self.val_trajectories_dir
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

        # Compositions
        train_compositions = surf.compositions_from_symbol_lists(
            self._train_symbol_lists, [num_template] * len(self._train_symbol_lists), self.species_names
        )
        sampled_symbol_lists = [a.get_chemical_symbols() for a in sampled_atoms]
        sampled_compositions = surf.compositions_from_symbol_lists(
            sampled_symbol_lists, [num_template] * len(sampled_atoms), self.species_names
        )

        energies_dict = {"Train": train_energies, "Sampled": sampled_energies}
        compositions_dict = {"Train": train_compositions, "Sampled": sampled_compositions}

        # Scalar metrics
        metrics = surf.energy_comparison_metrics(energies_dict, compositions_dict)
        for k, v in metrics.items():
            trainer.logger.log_metrics({f"val/{k}": v}, step=trainer.global_step)
        comp_w = surf.wasserstein_composition(train_compositions, sampled_compositions, use_first_species_only=True)
        trainer.logger.log_metrics({"val/composition_wasserstein": comp_w}, step=trainer.global_step)

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
            data_list = [{"atoms": a, "energy": float(sampled_energies[i]), "label": "VSSR-MC sample"} for i, a in enumerate(sampled_atoms)]
            energies_list = [float(sampled_energies[i]) for i in range(len(sampled_atoms))]
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

    def _log_figure(self, trainer: pl.Trainer, fig, key: str) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "log"):
            try:
                import wandb
                if isinstance(trainer.logger.experiment, wandb.run):
                    trainer.logger.experiment.log({key: wandb.Image(buf, format="png")}, step=trainer.global_step)
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
