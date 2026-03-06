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
        max_idx = max(self.number_to_element.keys())
        self._index_to_atomic_number = np.array(
            [self.number_to_element[i] for i in range(max_idx + 1)], dtype=np.int64
        )
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

    def get_energy_forces_atoms(
        self,
        atoms_list: list[ase.Atoms],
        show_progress: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MACE energies and forces for a list of ASE Atoms objects.

        Args:
            atoms_list: List of ASE Atoms objects.

        Returns:
            energies: Tensor of shape (N,) with total energies in eV.
            forces: Tensor of shape (sum(n_atoms_i), 3) with per-atom forces.
        """
        energies: list[float] = []
        forces: list[np.ndarray] = []
        calc = self.calculator
        iterator = tqdm(atoms_list, desc="Computing MACE energies", disable=not show_progress)
        with torch.inference_mode(mode=False), torch.enable_grad():
            for atoms in iterator:
                prev_calc = getattr(atoms, "calc", None)
                atoms.calc = calc
                # Request forces first; most ASE calculators cache energies/forces together.
                force = atoms.get_forces()
                energy = atoms.calc.results.get("energy", atoms.get_potential_energy())
                if isinstance(energy, torch.Tensor):
                    energy = energy.detach().cpu().item()
                energies.append(float(energy))
                forces.append(np.asarray(force, dtype=np.float32))
                atoms.calc = prev_calc
        energies_t = torch.tensor(energies, dtype=torch.float32)
        forces_t = torch.tensor(np.vstack(forces), dtype=torch.float32)
        return energies_t, forces_t

    def get_energy_atoms(
        self,
        atoms_list: list[ase.Atoms],
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Compute MACE energies for a list of ASE Atoms objects."""
        energies, _ = self.get_energy_forces_atoms(atoms_list, show_progress=show_progress)
        return energies

    def _batch_dict_to_atoms_list(self, batch: dict[str, Any]) -> list[ase.Atoms]:
        """Convert DSS/SchNetPack batch dict to list[ase.Atoms]."""
        z = batch.get("_atomic_numbers")
        pos = batch.get("_positions")
        n_atoms = batch.get("_n_atoms")
        if z is None or pos is None or n_atoms is None:
            raise ValueError("Batch must include _atomic_numbers, _positions and _n_atoms.")

        z_np = z.detach().cpu().numpy()
        pos_np = pos.detach().cpu().numpy()
        n_atoms_np = n_atoms.detach().cpu().numpy()
        cell = batch.get("_cell")
        cell_np = cell.detach().cpu().numpy() if cell is not None else None

        # If numbers are internal indices (e.g., 0/1/2), map to atomic numbers.
        if z_np.size > 0 and np.max(z_np) < self._index_to_atomic_number.shape[0]:
            z_np = self._index_to_atomic_number[z_np.astype(np.int64)]
        else:
            z_np = z_np.astype(np.int64)

        atoms_list = []
        idx = 0
        for i, n in enumerate(n_atoms_np):
            n = int(n)
            numbers = z_np[idx : idx + n]
            positions = pos_np[idx : idx + n]
            kwargs: dict[str, Any] = {}
            if cell_np is not None and cell_np.size > 0:
                if cell_np.ndim == 3:
                    kwargs["cell"] = cell_np[i]
                elif cell_np.ndim == 2 and cell_np.shape[0] >= 3 * (i + 1):
                    kwargs["cell"] = cell_np[i * 3 : (i + 1) * 3]
                kwargs["pbc"] = True
            atoms_list.append(ase.Atoms(numbers=numbers, positions=positions, **kwargs))
            idx += n
        return atoms_list

    def forward(self, inputs: Any) -> Any:
        """Forward pass supporting both list[Atoms] and DSS batch dict."""
        if isinstance(inputs, list):
            if len(inputs) == 0:
                return torch.empty(0, dtype=torch.float32)
            return self.get_energy_atoms(inputs, show_progress=False)
        if isinstance(inputs, dict):
            atoms_list = self._batch_dict_to_atoms_list(inputs)
            energies, forces = self.get_energy_forces_atoms(atoms_list, show_progress=False)
            device = inputs["_positions"].device
            dtype = inputs["_positions"].dtype
            energy_out = energies.to(device=device, dtype=dtype)
            forces_out = forces.to(device=device, dtype=dtype)
            return {"energy": energy_out, "forces": forces_out}
        raise TypeError(f"Unsupported input type for MACEEnergyModel.forward: {type(inputs)!r}")

    def get_energy(self, batch: Any, temp: torch.Tensor | None = None) -> torch.Tensor:
        """Compute MACE energies for a PyG Batch (optional; requires torch_geometric)."""
        atoms_list = self.batch_to_atoms_list(batch)
        energies = self.get_energy_atoms(atoms_list, show_progress=False)
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
