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
    """Load AgxOy dataset from directory (same layout as snowyflow). Returns (datamodule, template_atoms, z_confinement)."""
    from dss.data import agxoy as _agxoy
    return _agxoy.get_dataset_agxoy(
        data_path=data_path,
        template_system=template_system,
        mcmc_xyz_files=mcmc_xyz_files,
        z_confinement=z_confinement,
        path=path,
        batch_size=batch_size,
        num_train=num_train,
        num_val=num_val,
        num_workers=num_workers,
        cutoff=cutoff,
        neighbour_list=neighbour_list,
        split_file=split_file,
    )


def get_dataset(
    atoms,
    neighbour_list,
    path="dataset.db",
    repeats=[1, 2],
    mask_below_h=2.6,
    z_confinement=[2.5, 7.8],
    batch_size=32,
):
    import os
    import random

    import numpy as np
    import schnetpack.transform as trn
    from ase.calculators.singlepoint import SinglePointCalculator
    from schnetpack.data import ASEAtomsData, AtomsDataModule

    data = []
    for a in atoms:
        e = a.get_potential_energy()
        f = a.get_forces(apply_constraint=False).reshape(-1, 3)
        for r in repeats:
            a1 = a.copy()
            a1 = a1.repeat([r, r, 1])
            f1 = np.vstack([f] * r**2)
            a1.set_calculator(SinglePointCalculator(a1, energy=r**2 * e, forces=f1))

            data.append(a1)

    atoms = data
    

    print("=" * 10, "Creating dataset", "=" * 10)
    if os.path.exists(path):
        os.remove(path)
        if os.path.exists("split.npz"):
            os.remove("split.npz")

    property_list = []
    for a in atoms:
        e = a.get_potential_energy()
        f = a.get_forces().reshape(-1, 3)
        c = SinglePointCalculator(a, energy=e, forces=f)
        a.set_calculator(c)
        mask = np.zeros_like(f, dtype=bool)
        mask[a.get_positions()[:, 2] < mask_below_h] = True

        properties = {
            "energy": np.array([e]),
            "forces": f,
            "mask": mask,
            "z_confinement": z_confinement,
        }
        property_list.append(properties)

    dataset = ASEAtomsData.create(
        path,
        distance_unit="Ang",
        property_unit_dict={
            "energy": "eV",
            "forces": "eV/Ang",
            "mask": None,
            "z_confinement": None,
        },
    )
    dataset.add_systems(property_list, atoms)

    print("Number of reference calculations:", len(dataset))
    print("Available properties:")

    for p in dataset.available_properties:
        print("-", p)

    example = dataset[0]
    print("Properties of item in dataset:")

    for k, v in example.items():
        print("-", k, ":", v.shape)

    print("=" * 10, "Finished dataset", "=" * 10)

    dataset = AtomsDataModule(
        path,
        batch_size=batch_size,
        num_train=0.90,
        num_val=0.1,
        transforms=[
            neighbour_list,
            trn.CastTo32(),
        ],
        num_workers=32,
        pin_memory=True,
        split_file="split.npz",
    )
    dataset.prepare_data()
    dataset.setup()

    train_dataloader = dataset.train_dataloader()
    example = next(iter(train_dataloader))

    print("Properties of batch:")

    for k, v in example.items():
        print("-", k, ":", v.shape)

    print("idx:", example["_idx_m"])

    return dataset


def get_diffusion_model(
    cutoff=6.0,
    n_atom_basis=64,
    n_rbf=30,
    n_interactions=4,
    gated_blocks=4,
    beta_max=3.0,
    beta_min=1e-2,
    lr=1e-3,
    neighbour_list=None,
    potential_model_instance=None,
):
    import schnetpack as spk

    from dss.diffusion import VPDiffusion
    from dss.models import ConditionedScoreModel, Potential
    from dss.utils import TorchNeighborList

    if neighbour_list is None:
        neighbour_list = TorchNeighborList(cutoff)

    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
    representation = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )

    score_model = ConditionedScoreModel(
        representation, time_dim=2, gated_blocks=gated_blocks
    )

    if potential_model_instance is not None:
        potential = potential_model_instance
    else:
        pred_energy = spk.atomistic.Atomwise(
            n_in=representation.n_atom_basis, output_key="energy"
        )
        pred_forces = spk.atomistic.Forces(energy_key="energy", force_key="forces")
        pairwise_distance = spk.atomistic.PairwiseDistances()
        potential = Potential(
            representation=representation,
            input_modules=[pairwise_distance],
            output_modules=[pred_energy, pred_forces],
        )

    diffusion = VPDiffusion(
        score_model=score_model,
        potential_model=potential,
        neighbour_list=neighbour_list,
        beta_max=beta_max,
        beta_min=beta_min,
        optim_config={"lr": lr},
        scheduler_config={"factor": 0.90, "patience": 100},
    )

    return diffusion, neighbour_list


def get_energies_for_atoms(diffusion, atoms_list, num_template, z_confinement, batch_size=32):
    """Get energies for a list of ASE atoms using the diffusion model's potential.

    Converts atoms to batch format (mask, z_confinement), runs preprocess_batch
    and potential_model, returns per-structure energies.

    Args:
        diffusion: VPDiffusion module (must have potential_model).
        atoms_list: List of ASE Atoms (same num_template each).
        num_template: Number of template (fixed) atoms per structure.
        z_confinement: (z_min, z_max) or array of shape (2,).
        batch_size: Max structures per batch for inference.

    Returns:
        torch.Tensor of shape (len(atoms_list),) with total energy per structure (eV).
    """
    import numpy as np
    import schnetpack as spk
    import torch
    from ase import Atoms

    if getattr(diffusion, "potential_model", None) is None:
        raise ValueError("diffusion.potential_model is None; cannot compute energies")

    z = np.asarray(z_confinement, dtype=np.float32)
    if z.ndim == 1:
        z = z.reshape(1, 2)
    device = next(diffusion.parameters()).device

    all_energies = []
    for start in range(0, len(atoms_list), batch_size):
        chunk = atoms_list[start : start + batch_size]
        # Mask: first num_template True, rest False per structure
        mask_list = [
            np.vstack([
                np.ones((num_template, 3), dtype=bool),
                np.zeros((len(a) - num_template, 3), dtype=bool),
            ])
            for a in chunk
        ]
        mask = torch.tensor(np.vstack(mask_list), device=device)
        z_batch = torch.tensor(
            np.tile(z, (len(chunk), 1)),
            dtype=torch.float32,
            device=device,
        )
        converter = spk.interfaces.AtomsConverter(
            neighbor_list=None,
            additional_inputs={"mask": mask, "z_confinement": z_batch},
            device=str(device),
        )
        data = converter(chunk)
        if data["_pbc"].dim() > 1:
            data["_pbc"] = data["_pbc"].view(-1)
        # Converter may hand z_confinement per-atom; _split_batch expects (n_structures, 2)
        n_structures = len(chunk)
        data["z_confinement"] = z_batch.to(device).view(n_structures, 2)
        # Map converter keys to schnetpack properties if needed (preprocess_batch expects these)
        from schnetpack import properties as prop
        if prop.R not in data:
            data[prop.R] = data["_positions"]
        if prop.Z not in data:
            data[prop.Z] = data["_atomic_numbers"]
        if prop.n_atoms not in data and "_n_atoms" in data:
            data[prop.n_atoms] = data["_n_atoms"]
        if prop.idx_m not in data and "_idx_m" in data:
            data[prop.idx_m] = data["_idx_m"]
        if prop.cell not in data and "_cell" in data:
            data[prop.cell] = data["_cell"]
        if prop.pbc not in data and "_pbc" in data:
            data[prop.pbc] = data["_pbc"]
        # Force-response heads in SchNetPack require autograd during forward.
        with torch.set_grad_enabled(True):
            batch = diffusion.preprocess_batch(data, save_keys=[])
            out = diffusion.potential_model(batch)
        e = out["energy"]
        if e.dim() == 0:
            e = e.unsqueeze(0)
        elif e.size(0) != len(chunk):
            e = e.view(len(chunk), -1).sum(1)
        all_energies.append(e.cpu())
    return torch.cat(all_energies, dim=0)


def sample(
    diffusion,
    num_samples,
    template,
    symbols,
    z_confinement,
    num_steps=1000,
    eta=1e-2,
    postrelax_steps=100,
    return_trajectories=False,
    use_regressor_guidance=False,
):
    from collections import defaultdict
    import numpy as np
    import schnetpack as spk
    import torch
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    def to_atoms(batch_list):
        atoms = []
        for b in batch_list:
            a = Atoms(
                numbers=b["_atomic_numbers"].cpu().detach().numpy(),
                positions=b["_positions"].cpu().detach().numpy(),
                cell=b["_cell"].cpu().detach().numpy().reshape(3, 3),
                pbc=b["_pbc"].cpu().detach().numpy(),
            )
            try:
                e, f = b["energy"].cpu().item(), b["forces"].cpu().detach().numpy().reshape(-1, 3)
                a.calc = SinglePointCalculator(a, energy=e, forces=f)
            except Exception:
                # Some intermediate trajectory frames may not carry e/f predictions.
                pass

            atoms.append(a)
        return atoms

    dev = next(diffusion.parameters()).device
    z_conf = torch.tensor(np.asarray(z_confinement), dtype=torch.float32, device=dev)
    if z_conf.dim() == 1:
        z_conf = z_conf.unsqueeze(0)

    # Accept either:
    # - symbols: list[str] (same composition for all samples)
    # - symbols: list[list[str]] (per-sample composition)
    if (
        isinstance(symbols, (list, tuple))
        and len(symbols) > 0
        and isinstance(symbols[0], (list, tuple, np.ndarray))
    ):
        symbol_sets = [list(s) for s in symbols]
        if len(symbol_sets) != num_samples:
            raise ValueError(
                f"len(symbols)={len(symbol_sets)} must equal num_samples={num_samples} for per-sample symbols."
            )
    else:
        symbol_sets = [list(symbols)] * num_samples

    n_split = 64 if num_samples > 64 else max(1, num_samples)
    template_symbols = template.get_chemical_symbols()
    template_positions = template.get_positions()
    template_cell = template.get_cell()
    template_pbc = template.get_pbc()

    # Group by number of adsorbates so mask/converter shapes match.
    groups = defaultdict(list)
    for i, syms in enumerate(symbol_sets):
        groups[len(syms)].append(i)

    all_atoms = [None] * num_samples
    all_atoms_trajs = [None] * num_samples if return_trajectories else None

    for n_ads, indices in groups.items():
        mask = torch.tensor(
            np.vstack(
                [
                    np.ones((len(template), 3), dtype=bool),
                    np.zeros((n_ads, 3), dtype=bool),
                ]
            ),
            device=dev,
        )
        converter = spk.interfaces.AtomsConverter(
            neighbor_list=None,
            additional_inputs={"mask": mask, "z_confinement": z_conf},
            device=str(dev),
        )

        for start in range(0, len(indices), n_split):
            chunk_indices = indices[start : start + n_split]
            atoms_data = []
            for idx in chunk_indices:
                syms = symbol_sets[idx]
                all_symbols = template_symbols + syms
                positions = np.vstack((template_positions, np.zeros((len(syms), 3))))
                atoms_data.append(
                    Atoms(
                        all_symbols,
                        positions=positions,
                        cell=template_cell,
                        pbc=template_pbc,
                    )
                )

            data = converter(atoms_data)
            data["_pbc"] = data["_pbc"].view(-1)  # hack

            if return_trajectories:
                if use_regressor_guidance:
                    batch, traj_batch = diffusion.regressor_guidance_sample(
                        data,
                        num_steps=num_steps,
                        save_traj=True,
                        eta=eta,
                        postrelax_steps=postrelax_steps,
                    )
                else:
                    batch, traj_batch = diffusion.sample(
                        data,
                        num_steps=num_steps,
                        save_traj=True,
                    )

                chunk_trajs = [[] for _ in range(len(chunk_indices))]
                for b in traj_batch:
                    batch_list = diffusion._split_batch(b)
                    for j, item in enumerate(batch_list):
                        chunk_trajs[j].append(item)
                for j, batch_list in enumerate(chunk_trajs):
                    all_atoms_trajs[chunk_indices[j]] = to_atoms(batch_list)
            else:
                if use_regressor_guidance:
                    batch = diffusion.regressor_guidance_sample(
                        data,
                        num_steps=num_steps,
                        save_traj=False,
                        eta=eta,
                        postrelax_steps=postrelax_steps,
                    )
                else:
                    batch = diffusion.sample(
                        data,
                        num_steps=num_steps,
                        save_traj=False,
                    )

            # save final
            batch_list = diffusion._split_batch(batch, keep_ef=True)
            atoms = to_atoms(batch_list)
            for j, a in enumerate(atoms):
                all_atoms[chunk_indices[j]] = a

    if return_trajectories:
        return all_atoms, all_atoms_trajs
    return all_atoms
