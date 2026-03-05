"""Surface evaluation utilities for MACE energy calculations.

Standalone copy for DSS; no dependency on snowyflow.
Provides functions to precompute, save, load, and visualize MACE energies
for surface datasets (SrTiO3, AgxOy, and beyond).
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

logger = logging.getLogger(__name__)

from collections import Counter

from dss.data.constants.sto import (
    STO_BULK_ENERGIES,
    STO_REF_ELEMENT,
    STO_REF_FORMULA,
    STO_STOICS,
)


def get_atom_counts(atoms: object) -> Counter:
    """Get element counts from an atoms-like object.

    Accepts: ASE Atoms (get_chemical_symbols), dict/Series with 'counts' or
    'symbols', or an iterable of element symbols.

    Returns:
        Counter mapping element symbol -> count.
    """
    if hasattr(atoms, "get_chemical_symbols"):
        return Counter(atoms.get_chemical_symbols())
    if isinstance(atoms, (dict, pd.Series)) and "counts" in atoms:
        return Counter(atoms["counts"])
    if isinstance(atoms, (dict, pd.Series)) and "symbols" in atoms:
        return Counter(atoms["symbols"])
    return Counter(atoms)


def precompute_mace_energies(
    atoms_list: list,
    mace_model,
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute MACE energies for all structures in a dataset.

    Args:
        atoms_list: List of ASE Atoms objects from the dataset.
        mace_model: MACEEnergyModel instance.
        batch_size: Number of structures to process at a time (for logging).

    Returns:
        Tensor of shape (N,) with total energies in eV.
    """
    logger.info("Precomputing MACE energies for %d structures...", len(atoms_list))
    energies = mace_model.get_energy_atoms(atoms_list)
    logger.info(
        "MACE energies computed: mean=%.4f eV, std=%.4f eV, min=%.4f eV, max=%.4f eV",
        energies.mean().item(),
        energies.std().item(),
        energies.min().item(),
        energies.max().item(),
    )
    return energies


def save_mace_energies(
    energies: torch.Tensor,
    save_path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """Save precomputed MACE energies to disk.

    Args:
        energies: Tensor of shape (N,) with energies in eV.
        save_path: Path to save the .pt file.
        metadata: Optional metadata dict (e.g., model name, head, etc.).

    Returns:
        Path to the saved file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {"energies": energies}
    if metadata is not None:
        save_data["metadata"] = metadata
    torch.save(save_data, save_path)
    logger.info("MACE energies saved to %s", save_path)
    return save_path


def load_mace_energies(load_path: str | Path) -> tuple[torch.Tensor, dict | None]:
    """Load previously saved MACE energies.

    Args:
        load_path: Path to the .pt file.

    Returns:
        Tuple of (energies tensor, metadata dict or None).
    """
    data = torch.load(load_path, weights_only=False)
    energies = data["energies"]
    metadata = data.get("metadata")
    logger.info("Loaded %d MACE energies from %s", len(energies), load_path)
    return energies, metadata


def plot_energy_distribution(
    energies_dict: dict[str, torch.Tensor | np.ndarray],
    title: str = "Energy Distribution",
    xlabel: str = "Energy (eV)",
    per_atom: bool = False,
    num_atoms: int | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Plot energy distributions for comparison (e.g., dataset vs sampled).

    Creates a histogram comparing energy distributions from different sources.

    Args:
        energies_dict: Dict mapping label → energies. Each value is a tensor
            or array of energies. Example:
            {"Dataset (MACE)": dataset_energies, "Sampled (MACE)": sampled_energies}
        title: Plot title.
        xlabel: X-axis label.
        per_atom: If True and num_atoms is provided, normalize energies per atom.
        num_atoms: Number of atoms per structure (for per-atom normalization).
        save_path: Optional path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    rows: list[dict] = []
    for label, energies in energies_dict.items():
        if isinstance(energies, torch.Tensor):
            energies = energies.detach().cpu().numpy()
        energies = np.asarray(energies).flatten()
        if per_atom and num_atoms is not None and num_atoms > 0:
            energies = energies / num_atoms
        rows.append({"energy": energies, "label": label})
    if per_atom and num_atoms is not None and num_atoms > 0 and xlabel == "Energy (eV)":
        xlabel = "Energy (eV/atom)"

    energies_flat = np.concatenate([r["energy"] for r in rows])
    labels_flat = [r["label"] for r in rows for _ in range(len(r["energy"]))]
    df = pd.DataFrame({"energy": energies_flat, "label": labels_flat})

    sns.histplot(
        data=df,
        x="energy",
        hue="label",
        ax=ax,
        bins=100,
        alpha=0.4,
        stat="density",
        common_norm=False,
        common_bins=True,
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Energy distribution plot saved to %s", save_path)

    return fig


def compositions_from_symbol_lists(
    symbol_lists: list[list[str]],
    num_templates: list[int],
    species_names: tuple[str, ...] = ("Ag", "O"),
) -> list[tuple[int, ...]]:
    """Compute per-structure composition (adsorbate-only) from symbol lists.

    Args:
        symbol_lists: List of symbol lists, one per structure (full structure).
        num_templates: Number of template atoms per structure (same length as symbol_lists).
        species_names: Element symbols to count (e.g. ("Ag", "O") for AgxOy).

    Returns:
        List of tuples (count_0, count_1, ...) for adsorbate sites only.
    """
    out = []
    for syms, nt in zip(symbol_lists, num_templates, strict=True):
        adsorbate = syms[nt:]
        out.append(tuple(adsorbate.count(s) for s in species_names))
    return out


def plot_energy_per_composition(
    energies_dict: dict[str, np.ndarray | torch.Tensor],
    compositions_dict: dict[str, list[tuple[int, ...]]],
    title: str = "Energy distribution by composition (Train vs Sampled)",
    save_path: str | Path | None = None,
    max_compositions: int = 12,
) -> plt.Figure:
    """Plot energy distributions per composition bin, comparing Train and Sampled.

    Creates one subplot per composition that appears in the data; each subplot
    shows overlapping histograms of energies for that composition.

    Args:
        energies_dict: {"Train": energies, "Sampled": energies} (each 1d array).
        compositions_dict: {"Train": [(n_ag, n_o), ...], "Sampled": [...]} aligned with energies.
        title: Figure title.
        save_path: Optional path to save the figure.
        max_compositions: Max number of composition subplots (avoid huge figures).

    Returns:
        Matplotlib figure.
    """
    from collections import defaultdict

    comp_to_energies: dict[tuple[int, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for label, energies in energies_dict.items():
        if isinstance(energies, torch.Tensor):
            energies = energies.detach().cpu().numpy().flatten()
        else:
            energies = np.asarray(energies).flatten()
        comps = compositions_dict.get(label, [])
        if len(comps) != len(energies):
            continue
        for e, c in zip(energies, comps, strict=False):
            comp_to_energies[c][label].append(float(e))
    comps_sorted = sorted(comp_to_energies.keys())[:max_compositions]
    if not comps_sorted:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No composition data", ha="center", va="center")
        return fig
    n_plots = len(comps_sorted)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for idx, comp in enumerate(comps_sorted):
        ax = axes[idx]
        for label, elist in comp_to_energies[comp].items():
            if elist:
                ax.hist(elist, bins=30, alpha=0.5, label=label, density=True)
        ax.set_title(f"Composition {comp}")
        ax.set_xlabel("Energy (eV)")
        ax.legend()
    for idx in range(len(comps_sorted), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def energy_comparison_metrics(
    energies_dict: dict[str, np.ndarray | torch.Tensor],
    compositions_dict: dict[str, list[tuple[int, ...]]],
) -> dict[str, float]:
    """Summary statistics comparing sampled vs train energy distributions.

    Used for both STO and AgxOy (composition-agnostic). Computes pooled metrics
    (all energies) and per-composition metrics (averaged over compositions that
    have both train and sampled data).

    Args:
        energies_dict: {"Train": energies, "Sampled": energies} (each 1d array).
        compositions_dict: {"Train": [(n_0, n_1, ...), ...], "Sampled": [...]} aligned with energies.

    Returns:
        Dict with keys:
        - energy_wasserstein_pooled: Wasserstein distance between pooled distributions (eV).
        - energy_wasserstein_mean_per_comp: Mean of per-composition Wasserstein distances.
        - energy_mean_diff_pooled: mean(Sampled) - mean(Train) (eV).
        - energy_mean_diff_mean_per_comp: Mean of per-composition mean differences.
        - energy_std_ratio_pooled: std(Sampled) / std(Train), or 1.0 if train std is 0.
        - energy_std_ratio_mean_per_comp: Mean of per-composition std ratios.
    """
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        logger.warning("scipy not available; energy comparison metrics will be zeros")
        return {
            "energy_wasserstein_pooled": 0.0,
            "energy_wasserstein_mean_per_comp": 0.0,
            "energy_mean_diff_pooled": 0.0,
            "energy_mean_diff_mean_per_comp": 0.0,
            "energy_std_ratio_pooled": 1.0,
            "energy_std_ratio_mean_per_comp": 1.0,
        }

    def to_np(x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().flatten()
        return np.asarray(x).flatten()

    train_e = to_np(energies_dict.get("Train", np.array([])))
    sampled_e = to_np(energies_dict.get("Sampled", np.array([])))
    train_comp = compositions_dict.get("Train", [])
    sampled_comp = compositions_dict.get("Sampled", [])

    out: dict[str, float] = {}

    # Pooled metrics
    if len(train_e) > 0 and len(sampled_e) > 0:
        out["energy_wasserstein_pooled"] = float(wasserstein_distance(train_e, sampled_e))
        out["energy_mean_diff_pooled"] = float(np.mean(sampled_e) - np.mean(train_e))
        train_std = np.std(train_e)
        sampled_std = np.std(sampled_e)
        out["energy_std_ratio_pooled"] = float(sampled_std / train_std) if train_std > 0 else 1.0
    else:
        out["energy_wasserstein_pooled"] = 0.0
        out["energy_mean_diff_pooled"] = 0.0
        out["energy_std_ratio_pooled"] = 1.0

    # Per-composition: group energies by composition (same as plot_energy_per_composition)
    from collections import defaultdict

    comp_to_energies: dict[tuple[int, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for label, energies in energies_dict.items():
        e = to_np(energies)
        comps = compositions_dict.get(label, [])
        if len(comps) != len(e):
            continue
        for ei, c in zip(e, comps, strict=False):
            comp_to_energies[c][label].append(float(ei))

    w_per_comp: list[float] = []
    mean_diff_per_comp: list[float] = []
    std_ratio_per_comp: list[float] = []
    for comp, label_to_elist in comp_to_energies.items():
        train_list = label_to_elist.get("Train", [])
        sampled_list = label_to_elist.get("Sampled", [])
        if not train_list or not sampled_list:
            continue
        at = np.array(train_list)
        as_ = np.array(sampled_list)
        w_per_comp.append(wasserstein_distance(at, as_))
        mean_diff_per_comp.append(float(np.mean(as_) - np.mean(at)))
        t_std = np.std(at)
        s_std = np.std(as_)
        std_ratio_per_comp.append(float(s_std / t_std) if t_std > 0 else 1.0)

    if w_per_comp:
        out["energy_wasserstein_mean_per_comp"] = float(np.mean(w_per_comp))
        out["energy_mean_diff_mean_per_comp"] = float(np.mean(mean_diff_per_comp))
        out["energy_std_ratio_mean_per_comp"] = float(np.mean(std_ratio_per_comp))
    else:
        out["energy_wasserstein_mean_per_comp"] = 0.0
        out["energy_mean_diff_mean_per_comp"] = 0.0
        out["energy_std_ratio_mean_per_comp"] = 1.0

    return out


def wasserstein_composition(
    compositions_train: list[tuple[int, ...]],
    compositions_sampled: list[tuple[int, ...]],
    use_first_species_only: bool = True,
) -> float:
    """Wasserstein distance between train and sampled composition distributions.

    For binary (e.g. AgxOy), compositions are (n_Ag, n_O); n_O is determined by
    n_Ag when total adsorbate count is fixed, so we use the first component only
    by default.

    Args:
        compositions_train: List of (n_0, n_1, ...) for train structures.
        compositions_sampled: List of (n_0, n_1, ...) for sampled structures.
        use_first_species_only: If True, use only the first count (e.g. n_Ag) for 1d distance.

    Returns:
        Scalar Wasserstein distance.
    """
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        logger.warning("scipy not available; returning 0.0 for composition Wasserstein")
        return 0.0
    if use_first_species_only:
        a = np.array([c[0] for c in compositions_train if len(c) > 0], dtype=float)
        b = np.array([c[0] for c in compositions_sampled if len(c) > 0], dtype=float)
    else:
        a = np.array(compositions_train, dtype=float)
        b = np.array(compositions_sampled, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(wasserstein_distance(a, b))


def calculate_surface_energy(
    atoms: pd.Series | dict | object,
    energy: float,
    chem_pots: dict[str, float],
    offset_data: dict | None = None,
    bulk_energies: dict[str, float] | None = None,
) -> float:
    """Calculate surface energy for a given structure.

    Replicates logic from EnsembleNFFSurface.get_surface_energy.

    Args:
        atoms: Atoms object or dict/Series with 'symbols' or 'counts'.
               If it's an ASE Atoms object, we use get_chemical_symbols().
        energy: Total potential energy of the slab (eV).
        chem_pots: Chemical potentials (Delta mu) for elements (eV).
                   e.g. {'Sr': 0.0, 'O': 0.0, 'Ti': 0.0}
        offset_data: Optional override for bulk energies and stoichiometries.
                     Defaults to STO constants.
        bulk_energies: Optional override for bulk energies only (eV/atom or
                       eV/formula). When provided (e.g. from an MLIP), these
                       replace the bulk energies from offset_data or STO.
                       Keys should match: element symbols and ref formula
                       (e.g. 'Sr', 'Ti', 'O', 'SrTiO3' for STO).

    Returns:
        Surface energy (eV).
    """
    counts = get_atom_counts(atoms)

    if offset_data is None:
        if "Ag" in counts and counts.get(STO_REF_ELEMENT, 0) == 0:
            from dss.data.constants.agxoy import (
                AGXOY_BULK_ENERGIES,
                AGXOY_REF_ELEMENT,
                AGXOY_REF_FORMULA,
                AGXOY_STOICS,
            )

            bulk_energies_resolved = dict(AGXOY_BULK_ENERGIES)
            stoics = AGXOY_STOICS
            ref_formula = AGXOY_REF_FORMULA
            ref_element_resolved = AGXOY_REF_ELEMENT
        else:
            # Use default STO constants
            # Note: bulk energies in constants are in eV
            bulk_energies_resolved = dict(STO_BULK_ENERGIES)
            stoics = STO_STOICS
            ref_formula = STO_REF_FORMULA
            ref_element_resolved = STO_REF_ELEMENT
    else:
        bulk_energies_resolved = offset_data["bulk_energies"]
        stoics = offset_data["stoics"]
        ref_formula = offset_data["ref_formula"]
        ref_element_resolved = offset_data["ref_element"]

    if bulk_energies is not None:
        bulk_energies_resolved = {**bulk_energies_resolved, **bulk_energies}
    # 1. Start with slab energy
    surf_en = energy

    # 2. Subtract bulk energies (Reference State)
    # E_ref = N_Ti * E_bulk(SrTiO3) + sum((N_el - (n_el/n_Ti)*N_Ti) * E_bulk(el))
    n_ref = counts[ref_element_resolved]
    bulk_ref_en = n_ref * bulk_energies_resolved[ref_formula]
    for el, count in counts.items():
        if el != ref_element_resolved:
            stoi_ratio = stoics[el] / stoics[ref_element_resolved]
            excess = count - stoi_ratio * n_ref
            # Add contribution from elemental bulk
            bulk_ref_en += excess * bulk_energies_resolved.get(el, 0.0)
    surf_en -= bulk_ref_en

    # 3. Subtract chemical potential deviation (Delta mu)
    # This accounts for the variable conditions
    # term = sum((N_el - (n_el/n_Ti)*N_Ti) * mu_el)
    pot_term = 0.0
    for el, count in counts.items():
        if el != ref_element_resolved:
            stoi_ratio = stoics[el] / stoics[ref_element_resolved]
            excess = count - stoi_ratio * n_ref
            pot_term += excess * chem_pots.get(el, 0.0)
    surf_en -= pot_term

    return surf_en


def calculate_excess(
    atoms: object,
    target_element: str = "Sr",
    ref_element: str = "Ti",
    stoics: dict | None = None,
) -> float:
    """Calculate excess of target element relative to reference element.

    Gamma = N_target - N_ref * (stoic_target / stoic_ref)
    For SrTiO3: Gamma_Sr^Ti = N_Sr - N_Ti * (1/1) = N_Sr - N_Ti
    """
    if stoics is None:
        if ref_element == "Ag":
            from dss.data.constants.agxoy import AGXOY_STOICS

            stoics = AGXOY_STOICS
        else:
            stoics = STO_STOICS

    counts = get_atom_counts(atoms)

    n_target = counts[target_element]
    n_ref = counts[ref_element]
    ratio = stoics[target_element] / stoics[ref_element]

    return n_target - n_ref * ratio


def plot_surface_stability(
    data_list: list[dict],
    ref_chem_pots: list[dict],
    save_dir: str | Path | None = None,
    figsize: tuple = (6, 6),
    xlims: tuple[float, float] = (-6, 6),
    stripe_size: float = 50,
    stripe_linewidth: float = 2,
    energies: list[float] | None = None,
    offset_data: dict | None = None,
    bulk_energies: dict[str, float] | None = None,
    return_energies: bool = False,
    target_element: str = "Sr",
    ref_element: str = "Ti",
) -> list[plt.Figure] | tuple[list[plt.Figure], list[list[float]]]:
    """Generate surface stability plots (Omega_surf vs Gamma).

    Args:
        data_list: List of dicts containing:
            - 'atoms': ASE Atoms object (or symbols)
            - 'energy': Total energy (eV) — ignored if energies= is provided
            - 'label': str (optional, e.g. 'Sample', 'DL TiO2', etc.)
        ref_chem_pots: List of chemical potential conditions to plot.
            Each dict should have keys 'Sr', 'O', etc. and values in eV.
        save_dir: Directory to save plots.
        figsize: Figure size.
        xlims: X-axis limits.
        stripe_size: Size (length) of each VSSR-MC sample stripe in points^2.
            Smaller values give shorter stripes. Default 50.
        stripe_linewidth: Line width in points for each stripe. Default 2.
        energies: Optional list of total energies (eV), one per data_list entry.
            When provided (e.g. MLIP energies), used instead of d['energy'].
        offset_data: Optional override for bulk energies and stoichiometries
            when computing surface energy (passed to calculate_surface_energy).
        bulk_energies: Optional bulk energies (eV) to use when computing
            surface energy (e.g. from an MLIP); passed to calculate_surface_energy.
        return_energies: If True, return the energies used for the plots.

    Returns:
        List of generated Figures.
    """
    if energies is not None and len(energies) != len(data_list):
        raise ValueError(
            f"energies length ({len(energies)}) must match data_list length ({len(data_list)})"
        )
    if offset_data is None:
        if ref_element == "Ag":
            from dss.data.constants.agxoy import (
                AGXOY_BULK_ENERGIES,
                AGXOY_REF_ELEMENT,
                AGXOY_REF_FORMULA,
                AGXOY_STOICS,
            )

            offset_data = {
                "bulk_energies": dict(AGXOY_BULK_ENERGIES),
                "stoics": AGXOY_STOICS,
                "ref_formula": AGXOY_REF_FORMULA,
                "ref_element": AGXOY_REF_ELEMENT,
            }
        else:
            from dss.data.constants.sto import (
                STO_BULK_ENERGIES,
                STO_REF_ELEMENT,
                STO_REF_FORMULA,
                STO_STOICS,
            )

            offset_data = {
                "bulk_energies": dict(STO_BULK_ENERGIES),
                "stoics": STO_STOICS,
                "ref_formula": STO_REF_FORMULA,
                "ref_element": STO_REF_ELEMENT,
            }

    figs = []

    # Pre-calculate Gamma for all structures
    # We assume Sr vs Ti for STO plots
    gammas = []
    stoics = offset_data["stoics"] if offset_data is not None else None
    for d in data_list:
        g = calculate_excess(d["atoms"], target_element, ref_element, stoics=stoics)
        gammas.append(g)

    for i, pot in enumerate(ref_chem_pots):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(xlims)
        # Calculate Omega for all structures under this potential
        omegas = []
        labels = []

        # We want to group by Gamma to create the "boxplot" style for samples
        # and simple markers for reference structures

        sample_omegas = []
        sample_gammas = []

        ref_points = []  # (gamma, omega, label, marker, color)

        for j, d in enumerate(data_list):
            ej = energies[j] if energies is not None else d["energy"]
            omega = calculate_surface_energy(
                d["atoms"],
                ej,
                pot,
                offset_data=offset_data,
                bulk_energies=bulk_energies,
            )
            label = d.get("label", "Sample")

            if "Sample" in label or "VSSR" in label:
                sample_omegas.append(omega)
                sample_gammas.append(gammas[j])
            else:
                # Store reference points to plot on top
                marker = d.get("marker", "X")
                color = d.get("color", "black")
                s = d.get("size", 200)
                ref_points.append(
                    {
                        "gamma": gammas[j],
                        "omega": omega,
                        "label": label,
                        "marker": marker,
                        "color": color,
                        "s": s,
                    }
                )

        # Plot Samples (VSSR-MC)
        # The reference plots show "strips" for samples at each integer Gamma.
        # We can use a custom approach or scatter with transparency/custom marker shape.
        # The image shows horizontal lines (like a 1D scatter/strip plot).
        if sample_gammas:
            # Create a dataframe for easy plotting if needed, or just scatter
            # "VSSR-MC sample" uses a light orange horizontal line marker
            ax.scatter(
                sample_gammas,
                sample_omegas,
                c="orange",
                alpha=0.4,
                marker="_",
                s=stripe_size,
                linewidths=stripe_linewidth,
                label="VSSR-MC sample",
            )

        # Plot References
        for pt in ref_points:
            ax.scatter(
                [pt["gamma"]],
                [pt["omega"]],
                label=pt["label"],
                marker=pt["marker"],
                s=pt["s"],
                edgecolors="k",
                c=pt["color"],
                zorder=10,  # Put on top
            )

        # Min Omega_surf line (optional, based on plot)
        # Find min omega across ALL data for this potential?
        all_omegas = sample_omegas + [p["omega"] for p in ref_points]
        if all_omegas:
            min_omega = min(all_omegas)
            ax.axhline(
                min_omega,
                color="#D81B60",
                linestyle="--",
                alpha=0.8,
                label=r"Min. $\Omega_{\text{surf}}$",
            )
            # a text label for the value
            ax.text(
                np.mean(ax.get_xlim()),
                min_omega,
                f"{min_omega:.1f} eV",
                va="bottom",
                ha="center",
                color="black",  # label color logic
                fontsize=8,
                backgroundcolor="white",  # basic readability
            )

        # Labels and Title
        mu_str = ", ".join([rf"$\mu_{{{k}}} = {v}$ eV" for k, v in pot.items() if v != 0])
        if not mu_str:
            mu_str = r"$\mu = 0$ eV"

        ax.set_title(mu_str, fontsize=14)
        ax.set_xlabel(
            rf"$\Gamma_{{\text{{{target_element}}}}}^{{\text{{{ref_element}}}}}$ [# {target_element} - # {ref_element}]",
            fontsize=14,
        )
        ax.set_ylabel(r"$\Omega_{\text{surf}}$ [eV]", fontsize=14)

        # Legend
        # Deduplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles, strict=False))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=12,
            frameon=True,
            framealpha=1,
            edgecolor="black",
        )

        # Style
        ax.tick_params(direction="in", labelsize=12, width=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        plt.tight_layout()

        if save_dir:
            path = Path(save_dir) / f"surface_stability_{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved plot to %s", path)

        figs.append(fig)
        if return_energies:
            energies.append(all_omegas)

    return figs


def get_top_k_per_chem_pot(
    data_list,
    chem_pots_list,
    k=5,
    bulk_energies=None,
    offset_data=None,
    target_element="Sr",
    ref_element="Ti",
):
    """Return for each chem_pot the top k structures with lowest surface energy (Ω_surf).

    Returns:
        list of length len(chem_pots_list). Each element is a list of up to k dicts with
        keys: index, omega, gamma, data (original data_list entry).
    """
    stoics = offset_data["stoics"] if offset_data is not None else None
    gammas = [
        calculate_excess(d["atoms"], target_element, ref_element, stoics=stoics) for d in data_list
    ]
    results = []
    for pot in chem_pots_list:
        omegas = []
        for j, d in enumerate(data_list):
            ej = d["energy"]
            omega = calculate_surface_energy(
                d["atoms"],
                ej,
                pot,
                offset_data=offset_data,
                bulk_energies=bulk_energies,
            )
            omegas.append((j, omega))
        omegas.sort(key=lambda x: x[1])
        top_k = []
        for j, omega in omegas[:k]:
            top_k.append(
                {
                    "index": j,
                    "omega": omega,
                    "gamma": gammas[j],
                    "data": data_list[j],
                }
            )
        results.append(top_k)
    return results
