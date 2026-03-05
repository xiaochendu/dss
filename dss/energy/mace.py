"""MACE-based energy model for surface systems (AgxOy and beyond).

Copied from snowyflow; wraps the pre-trained MACE MH-1 foundation model
as an energy calculator with get_energy_atoms(list[ase.Atoms]) for DSS surface eval.
"""

import logging
from typing import Any

import ase
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

# eV/K (Boltzmann constant)
K_B = 8.617333e-5


class MACEEnergyModel(nn.Module):
    """Pre-trained MACE energy model for surface systems.

    Uses MACE MH-1 (or other MACE foundation models) to compute total
    potential energies for atomic structures. Energies are extensive
    (total energy in eV, sum of per-atom contributions).

    The model wraps the MACE ASE calculator interface and provides
    get_energy_atoms(atoms_list) for use in DSS surface eval.
    """

    def __init__(
        self,
        model: str = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
        head: str = "omat_pbe",
        device: str = "cuda",
        default_dtype: str = "float64",
        dispersion: bool = True,
        enable_cueq: bool = False,
        number_to_element: dict[int, int] | None = None,
    ) -> None:
        """Initialize the MACE energy model.

        Args:
            model: Model name or path. For MH-1, use the HuggingFace URL or
                local path to .model file. Defaults to MACE MH-1.
            head: Model head to use. Options: "omat_pbe", "oc20", "omol", etc.
                Defaults to "omat_pbe" (recommended for surfaces).
            device: Device for computation ("cuda" or "cpu").
            default_dtype: Default floating point precision ("float32" or "float64").
            dispersion: Whether to add D3 dispersion correction. Requires torch-dftd.
            enable_cueq: Whether to use cuEquivariance for acceleration.
            number_to_element: Mapping from internal index to atomic number.
                If None, uses AgxOy default: {0: 47 (Ag), 1: 8 (O), 2: 1 (mask)}.
        """
        super().__init__()
        self._device_str = device
        self._model_path = model
        self._head = head
        self._default_dtype = default_dtype
        self._dispersion = dispersion
        self._enable_cueq = enable_cueq

        if number_to_element is not None:
            self.number_to_element = number_to_element
        else:
            from dss.data.constants.agxoy import mask_index, number_to_element as agxoy_map

            self.number_to_element = {k: v for k, v in agxoy_map.items() if k <= mask_index}
        self._calculator = None

    @property
    def calculator(self):
        """Lazy-load the MACE calculator on first use."""
        if self._calculator is None:
            self._calculator = self._load_calculator()
        return self._calculator

    def _load_calculator(self):
        """Load the MACE ASE calculator."""
        from mace.calculators import mace_mp

        logger.info(
            "Loading MACE model: %s (head=%s, dispersion=%s, cueq=%s)",
            self._model_path,
            self._head,
            self._dispersion,
            self._enable_cueq,
        )

        try:
            calc = mace_mp(
                model=self._model_path,
                default_dtype=self._default_dtype,
                device=self._device_str,
                dispersion=self._dispersion,
                head=self._head,
                enable_cueq=self._enable_cueq,
            )
        except Exception:  # pylint: disable=broad-except
            if self._enable_cueq:
                logger.warning("cuEquivariance failed, falling back to standard MACE evaluation.")
                calc = mace_mp(
                    model=self._model_path,
                    default_dtype=self._default_dtype,
                    device=self._device_str,
                    dispersion=self._dispersion,
                    head=self._head,
                    enable_cueq=False,
                )
            else:
                raise

        logger.info("MACE model loaded successfully.")
        return calc

    def get_energy_atoms(
        self,
        atoms_list: list[ase.Atoms],
    ) -> torch.Tensor:
        """Compute MACE energies for a list of ASE Atoms objects.

        Args:
            atoms_list: List of ASE Atoms objects.

        Returns:
            Tensor of shape (N,) containing total energies in eV.
        """
        energies = []
        calc = self.calculator
        with torch.inference_mode(mode=False):
            with torch.enable_grad():
                for atoms in tqdm(atoms_list, desc="Computing MACE energies"):
                    atoms_copy = atoms.copy()
                    atoms_copy.calc = calc
                    energy = atoms_copy.get_potential_energy()
                    if isinstance(energy, torch.Tensor):
                        energy = energy.detach().cpu().item()
                    energies.append(float(energy))
                    atoms_copy.calc = None
                    del atoms_copy
                    getattr(calc, "clear_cache", lambda: None)()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        return torch.tensor(energies, dtype=torch.float32)

    def forward(self, atoms_list: list[ase.Atoms]) -> torch.Tensor:
        """Forward pass: compute MACE energies for a list of ASE Atoms."""
        return self.get_energy_atoms(atoms_list)

    def get_energy(self, batch: Any, temp: torch.Tensor | None = None) -> torch.Tensor:
        """Compute MACE energies for a PyG Batch (optional; requires torch_geometric)."""
        atoms_list = self.batch_to_atoms_list(batch)
        energies = self.get_energy_atoms(atoms_list)
        if temp is not None:
            dtype = temp.dtype
            logf_t = -energies.to(device=self._device_str, dtype=dtype) / (
                K_B * temp.detach().to(device=self._device_str, dtype=dtype)
            )
            return logf_t
        return energies

    def batch_to_atoms_list(self, batch: Any) -> list[ase.Atoms]:
        """Convert a PyG Batch to a list of ASE Atoms (optional; requires torch_geometric)."""
        batch_size = batch.num_graphs
        batch_idx = batch.batch.cpu()
        numbers = batch["numbers_data"].cpu().numpy()
        if hasattr(batch, "positions") and batch.positions is not None:
            positions = batch["positions"].cpu().numpy()
        else:
            positions = batch["pos"].cpu().numpy()
        has_cell = False
        cells = None
        for key in ("lattice", "cell"):
            if hasattr(batch, key) and getattr(batch, key) is not None:
                cells = batch[key].cpu().numpy()
                has_cell = True
                break

        atoms_list = []
        for i in range(batch_size):
            node_mask = (batch_idx == i).numpy()
            atom_numbers = numbers[node_mask]
            bad = [int(n) for n in atom_numbers if int(n) not in self.number_to_element]
            if bad:
                raise ValueError(
                    "Batch contains mask or unknown species index; cannot compute energy. "
                    f"Unknown indices: {bad}"
                )
            atomic_numbers = np.array([self.number_to_element[int(n)] for n in atom_numbers])
            atom_positions = positions[node_mask]
            if has_cell and cells is not None:
                if cells.ndim == 2 and cells.shape[0] == batch_size * 3:
                    cell = cells[i * 3 : (i + 1) * 3]
                elif cells.ndim == 3:
                    cell = cells[i]
                else:
                    cell = cells[i] if cells.shape[0] == batch_size else cells
                atoms = ase.Atoms(
                    numbers=atomic_numbers,
                    positions=atom_positions,
                    cell=cell,
                    pbc=True,
                )
            else:
                atoms = ase.Atoms(numbers=atomic_numbers, positions=atom_positions)
            atoms_list.append(atoms)
        return atoms_list
