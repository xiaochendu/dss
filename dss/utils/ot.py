import numpy as np
import ot
import torch


def solve_ot_assignment(
    noise_positions: torch.Tensor,
    data_positions: torch.Tensor,
    lattice: torch.Tensor,
    periodic: bool = True,
    species_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Solve the Optimal Transport assignment problem to find the straightest paths.
    Ported from snowyflow. Supports PyTorch tensors.

    Args:
        noise_positions: Initial (prior) positions of adsorbate atoms (N, 3).
        data_positions: Target ground-truth positions of adsorbate atoms (N, 3).
        lattice: Unit cell lattice matrix (3, 3).
        periodic: If True, uses Minimum Image Convention (MIC) for distances.
        species_labels: Optional labels (N,) identifying the species of each atom.
            If provided, OT is solved independently for each species to preserve
            composition conditioning.

    Returns:
        reordered_noise: Noise positions permuted to match the data assignment.
    """
    if noise_positions.shape[0] == 0:
        return noise_positions

    device = noise_positions.device
    num_atoms = noise_positions.shape[0]
    reordered_noise = torch.zeros_like(noise_positions)

    # Convert to numpy for POT library
    noise_np = noise_positions.detach().cpu().numpy()
    data_np = data_positions.detach().cpu().numpy()
    lattice_np = lattice.detach().cpu().numpy()
    
    if species_labels is not None:
        species_np = species_labels.detach().cpu().numpy()
        unique_species = np.unique(species_np)
    else:
        species_np = None
        unique_species = [None]

    for species in unique_species:
        if species is not None:
            idx = np.where(species_np == species)[0]
        else:
            idx = np.arange(num_atoms)

        if len(idx) == 0:
            continue

        curr_noise = noise_np[idx]
        curr_data = data_np[idx]

        # 1. Compute Distance/Cost Matrix
        if periodic:
            # Calculate periodic pairwise distance matrix using MIC
            diffs = curr_noise[:, None, :] - curr_data[None, :, :]
            # Using numpy inv for the cost matrix calculation
            inv_lattice = np.linalg.inv(lattice_np)
            frac_diffs = diffs @ inv_lattice
            frac_diffs[:, :, :2] -= np.round(frac_diffs[:, :, :2])  # Periodic in XY
            mic_diffs = frac_diffs @ lattice_np
            loss_matrix = np.sum(mic_diffs**2, axis=-1)
        else:
            loss_matrix = ot.dist(curr_noise, curr_data)

        # 2. Solve Assignment (Earth Mover's Distance)
        # Uniform weights for both distributions
        plan = ot.emd([], [], loss_matrix)
        permute_index = plan.argmax(axis=0)

        # 3. Apply Permutation
        reordered_noise[idx] = torch.from_tensor(curr_noise[permute_index]).to(device) if hasattr(torch, "from_tensor") else torch.from_numpy(curr_noise[permute_index]).to(device)

    return reordered_noise
