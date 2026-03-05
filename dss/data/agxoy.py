"""Load AgxOy surface dataset from the same directory layout as snowyflow.

Directory should contain:
- template_{template_system}.xyz (e.g. template_111_c4x8.xyz)
- agox_sample_AgxOy_*.xyz (one or more MCMC output files)

Compatible with data at e.g. /mnt/data0/dux/vssr_fm_working/data/agox_AgxOy_structures.
"""

from pathlib import Path

import numpy as np
import schnetpack.transform as trn
from ase import io as ase_io
from ase.calculators.singlepoint import SinglePointCalculator
from schnetpack.data import ASEAtomsData, AtomsDataModule

from dss.utils import TorchNeighborList


def _read_mcmc_xyz(xyz_file: Path) -> list:
    """Load structures from a multi-frame XYZ; deduplicate by (len, positions)."""
    atoms_list_full = ase_io.read(str(xyz_file), index=":")
    atoms_list = []
    for atoms in atoms_list_full:
        for atoms_compare in atoms_list:
            if len(atoms) == len(atoms_compare):
                if np.allclose(atoms.get_positions(), atoms_compare.get_positions()):
                    break
        else:
            atoms_list.append(atoms)
    return atoms_list


def _match_system(atoms, template_atoms_dict: dict) -> str:
    """Return system key if atoms starts with that template (same elements and positions)."""
    elements = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    for system, template_atoms in template_atoms_dict.items():
        if len(atoms) < len(template_atoms):
            continue
        elements_template = template_atoms.get_atomic_numbers()
        positions_template = template_atoms.get_positions()
        length = len(elements_template)
        if np.all(elements[:length] == elements_template):
            if np.allclose(positions[:length], positions_template):
                return system
    raise ValueError("No matching system found.")


def get_dataset_agxoy(
    data_path,
    template_system="111_c4x8",
    mcmc_xyz_files=None,
    z_confinement=None,
    path="dataset.db",
    batch_size=32,
    num_train=0.90,
    num_val=0.1,
    num_workers=0,
    cutoff=6.0,
    neighbour_list=None,
    split_file="split.npz",
):
    """Build schnetpack dataset and datamodule from AgxOy XYZ directory (same layout as snowyflow).

    Args:
        data_path: Directory containing template_*.xyz and agox_sample_AgxOy_*.xyz.
        template_system: Template name (default "111_c4x8") -> template_111_c4x8.xyz.
        mcmc_xyz_files: List of MCMC XYZ filenames (relative to data_path). If None, glob agox_sample_AgxOy_*.xyz.
        z_confinement: [z_min, z_max] for adsorbate layer. If None, derived from template and slab z.
        path: Path for the created .db file.
        batch_size: Batch size for dataloaders.
        num_train: Fraction of data for training (default 0.9).
        num_val: Fraction for validation (default 0.1).
        num_workers: DataLoader num_workers (default 0 for stability).
        cutoff: Cutoff for neighbour list (used if neighbour_list is None).
        neighbour_list: Schnetpack transform for neighbour list. If None, TorchNeighborList(cutoff) is used.
        split_file: File for train/val split (default split.npz).

    Returns:
        (datamodule, template_atoms, z_confinement_used)
        - datamodule: schnetpack AtomsDataModule (prepare_data and setup already called).
        - template_atoms: ASE Atoms of the template (for sampling).
        - z_confinement_used: [z_min, z_max] used for all structures.
    """
    import os

    root = Path(data_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Data path is not a directory: {root}")

    # Load template
    template_path = root / f"template_{template_system}.xyz"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    template_atoms = ase_io.read(str(template_path))
    template_atoms_dict = {template_system: template_atoms}
    num_template = len(template_atoms)

    # Resolve MCMC XYZ files
    if mcmc_xyz_files is None:
        mcmc_xyz_files = sorted(root.glob("agox_sample_AgxOy_*.xyz"))
        mcmc_xyz_files = [f.name for f in mcmc_xyz_files]
    if not mcmc_xyz_files:
        raise FileNotFoundError(f"No MCMC XYZ files found in {root}")

    # Load all structures and filter by template match
    data_atoms = []
    for mcmc_xyz in mcmc_xyz_files:
        p = root / mcmc_xyz
        if not p.exists():
            continue
        for atoms in _read_mcmc_xyz(p):
            try:
                _match_system(atoms, template_atoms_dict)
            except ValueError:
                continue
            data_atoms.append(atoms)

    if not data_atoms:
        raise ValueError(
            f"No structures matched template (len={num_template}) in {root}. "
            "Check that MCMC XYZ frames have template as first N atoms."
        )

    # z_confinement: derive if not provided
    if z_confinement is None:
        template_max_z = template_atoms.positions[:, 2].max()
        slab_max_z = max(a.positions[:, 2].max() for a in data_atoms)
        z_confinement = [float(template_max_z + 0.5), float(slab_max_z + 0.5)]
    else:
        z_confinement = list(z_confinement)

    # Ensure each atoms has calculator (energy/forces); use 0 if missing (e.g. plain XYZ)
    for a in data_atoms:
        if a.calc is None:
            n = len(a)
            a.set_calculator(
                SinglePointCalculator(
                    a,
                    energy=0.0,
                    forces=np.zeros((n, 3)),
                )
            )

    # Build property_list and atoms for schnetpack (template-based mask)
    property_list = []
    for a in data_atoms:
        e = a.get_potential_energy()
        f = a.get_forces(apply_constraint=False).reshape(-1, 3)
        n = len(a)
        mask = np.zeros((n, 3), dtype=bool)
        mask[:num_template, :] = True
        mask[num_template:, :] = False
        property_list.append({
            "energy": np.array([e], dtype=np.float32),
            "forces": f.astype(np.float32),
            "mask": mask,
            "z_confinement": np.array(z_confinement, dtype=np.float32),
        })

    # Create DB (remove existing so split is fresh)
    path = Path(path)
    if path.exists():
        os.remove(path)
    if Path(split_file).exists():
        os.remove(split_file)

    print("=" * 10, "Creating AgxOy dataset", "=" * 10)
    print(f"Template: {template_path} (N={num_template})")
    print(f"Structures: {len(data_atoms)}")
    print(f"z_confinement: {z_confinement}")

    dataset = ASEAtomsData.create(
        str(path),
        distance_unit="Ang",
        property_unit_dict={
            "energy": "eV",
            "forces": "eV/Ang",
            "mask": None,
            "z_confinement": None,
        },
    )
    dataset.add_systems(property_list, data_atoms)

    if neighbour_list is None:
        neighbour_list = TorchNeighborList(cutoff)

    datamodule = AtomsDataModule(
        str(path),
        batch_size=batch_size,
        num_train=num_train,
        num_val=num_val,
        transforms=[
            neighbour_list,
            trn.CastTo32(),
        ],
        num_workers=num_workers,
        pin_memory=(num_workers == 0),
        split_file=split_file,
    )
    datamodule.prepare_data()
    datamodule.setup()

    print("=" * 10, "Finished AgxOy dataset", "=" * 10)
    return datamodule, template_atoms, z_confinement
