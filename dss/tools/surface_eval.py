"""Surface evaluation utilities for MACE energy calculations.

Provides functions to precompute, save, load, and visualize MACE energies
for surface datasets (SrTiO3 and beyond).
"""

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path as _Path
from scipy.stats import gaussian_kde as _gaussian_kde

logger = logging.getLogger(__name__)

from collections import Counter

from dss.data.constants.sto import (
    STO_BULK_ENERGIES,
    STO_REF_ELEMENT,
    STO_REF_FORMULA,
    STO_STOICS,
)


def get_atom_counts(atoms: object) -> Counter:
    """Get element counts from an atoms-like object."""
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
    """Compute MACE energies for all structures in a dataset."""
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
    """Save precomputed MACE energies to disk."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {"energies": energies}
    if metadata is not None:
        save_data["metadata"] = metadata
    torch.save(save_data, save_path)
    logger.info("MACE energies saved to %s", save_path)
    return save_path


def load_mace_energies(load_path: str | Path) -> tuple[torch.Tensor, dict | None]:
    """Load previously saved MACE energies."""
    data = torch.load(load_path, weights_only=False)
    energies = data["energies"]
    metadata = data.get("metadata")
    logger.info("Loaded %d MACE energies from %s", len(energies), load_path)
    return energies, metadata


def _resolve_color(label: str) -> str:
    """Map a distribution label to the project color cycle."""
    l = label.lower()
    if "train" in l:
        return "C5"
    if any(k in l for k in ("snowy", "dfm", "flow", "sampled")):
        return "C0"
    if any(k in l for k in ("dss", "vp", "diffusion")):
        return "C1"
    return "C0"


def plot_energy_distribution(
    energies_dict: dict[str, torch.Tensor | np.ndarray],
    title: str = "Energy Distribution",
    xlabel: str = "Energy (eV)",
    per_atom: bool = False,
    num_atoms: int | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (3.375, 2.5),
    bins: int = 50,
    xlims: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot energy distributions using KDE curves (journal style)."""
    # Load shared matplotlibrc if available
    _rc = _Path.home() / "matplotlibrc" / "matplotlibrc"
    if _rc.exists():
        mpl.rc_file(str(_rc))

    FONTSIZE = 8
    fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")

    processed_energies = {}
    all_vals = []
    for label, energies in energies_dict.items():
        if isinstance(energies, torch.Tensor):
            energies = energies.detach().cpu().numpy()
        energies = np.asarray(energies).flatten()
        if per_atom and num_atoms is not None and num_atoms > 0:
            energies = energies / num_atoms
        processed_energies[label] = energies
        all_vals.extend(energies.tolist())

    if per_atom and num_atoms is not None and num_atoms > 0 and xlabel == "Energy (eV)":
        xlabel = "Energy (eV/atom)"

    if xlims is None:
        min_e, max_e = float(np.min(all_vals)), float(np.max(all_vals))
    else:
        min_e, max_e = xlims
    xs = np.linspace(min_e, max_e, 400)
    ax.set_xlim(min_e, max_e)

    # Plot Train last so it sits on top; collect non-Train first
    order = [k for k in processed_energies if k != "Train"] + (
        ["Train"] if "Train" in processed_energies else []
    )
    for label in order:
        energies = processed_energies[label]
        if len(energies) < 2:
            continue
        color = _resolve_color(label)
        lw = 1.0 if label == "Train" else 1.5
        ls = "--" if label == "Train" else "-"
        alpha_fill = 0.10 if label == "Train" else 0.15
        try:
            kde = _gaussian_kde(energies)
            ys = kde(xs)
        except Exception:
            continue
        ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, label=label)
        ax.fill_between(xs, ys, alpha=alpha_fill, color=color, linewidth=0)

    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel("Density", fontsize=FONTSIZE)
    if title:
        ax.set_title(title, fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    leg = ax.legend(frameon=True, fontsize=FONTSIZE)
    leg.get_frame().set_boxstyle("Square", pad=0)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        logger.info("Energy distribution plot saved to %s", save_path)

    return fig


def compositions_from_symbol_lists(
    symbol_lists: list[list[str]],
    num_templates: list[int],
    species_names: tuple[str, ...] = ("Ag", "O"),
) -> list[tuple[int, ...]]:
    """Compute per-structure composition (adsorbate-only) from symbol lists."""
    out = []
    for syms, nt in zip(symbol_lists, num_templates, strict=True):
        adsorbate = syms[nt:]
        out.append(tuple(adsorbate.count(s) for s in species_names))
    return out


def plot_energy_per_composition(
    energies_dict: dict[str, np.ndarray | torch.Tensor],
    compositions_dict: dict[str, list[tuple[int, ...]]],
    title: str = "Energy distribution by composition",
    save_path: str | Path | None = None,
    max_compositions: int = 12,
    bins: int = 30,
    xlims: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot energy distributions per composition bin with standardized full-width."""
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
            
    # Determine global set of compositions present in either Train or Model
    all_comps = set()
    for label in energies_dict.keys():
        all_comps.update(compositions_dict.get(label, []))
    
    comps_sorted = sorted(list(all_comps))[:max_compositions]
    if not comps_sorted:
        fig, ax = plt.subplots(1, 1, figsize=(6.7, 4))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No composition data", ha="center", va="center")
        return fig
        
    n_plots = len(comps_sorted)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Full-width standardized sizing (each panel ~2.2")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.7, 2.2 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = {"Train": "#495567", "snowyflow": "#6465E9", "dss": "#4BA3E3", "Sampled": "#E8A23B"}

    for idx, comp in enumerate(comps_sorted):
        ax = axes[idx]
        
        comp_vals = []
        for elist in comp_to_energies[comp].values():
            comp_vals.extend(elist)
        if not comp_vals: continue
        
        if xlims is None:
            min_e, max_e = min(comp_vals), max(comp_vals)
        else:
            min_e, max_e = xlims
            ax.set_xlim(xlims)
            
        bin_edges = np.linspace(min_e, max_e, bins + 1)

        for label, elist in comp_to_energies[comp].items():
            if elist:
                sns.histplot(
                    elist,
                    ax=ax,
                    bins=bin_edges,
                    label=label,
                    color=colors.get(label),
                    alpha=0.4,
                    stat="density",
                    element="step",
                )
        
        # LaTeX chemical formula title
        ax.set_title(rf"$\text{{Ag}}_{{{comp[0]}}}\text{{O}}_{{{comp[1]}}}$", fontsize=10)
        ax.set_xlabel("Energy (eV)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=8, frameon=False)
            
    for idx in range(len(comps_sorted), len(axes)):
        axes[idx].set_visible(False)
        
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def energy_comparison_metrics(
    energies_dict: dict[str, np.ndarray | torch.Tensor],
    compositions_dict: dict[str, list[tuple[int, ...]]],
) -> dict[str, float]:
    """Summary statistics comparing sampled vs train energy distributions."""
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
    
    # Use the first key that is NOT 'Train' as the sampled distribution.
    sampled_label = None
    for label in energies_dict.keys():
        if label != "Train":
            sampled_label = label
            break
            
    if sampled_label is None:
        return {
            "energy_wasserstein_pooled": 0.0,
            "energy_wasserstein_mean_per_comp": 0.0,
            "energy_mean_diff_pooled": 0.0,
            "energy_mean_diff_mean_per_comp": 0.0,
            "energy_std_ratio_pooled": 1.0,
            "energy_std_ratio_mean_per_comp": 1.0,
        }
        
    sampled_e = to_np(energies_dict.get(sampled_label, np.array([])))
    train_comp = compositions_dict.get("Train", [])
    sampled_comp = compositions_dict.get(sampled_label, [])

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

    # Per-composition: group energies by composition
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
        sampled_list = label_to_elist.get(sampled_label, [])
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
    """Wasserstein distance between train and sampled composition distributions."""
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
    """Calculate surface energy for a given structure."""
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
            # STO_REF_ELEMENT already imported at module level — don't re-import
            # locally or Python will treat it as a local variable throughout the function
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
    
    surf_en = energy
    n_ref = counts[ref_element_resolved]
    bulk_ref_en = n_ref * bulk_energies_resolved[ref_formula]
    for el, count in counts.items():
        if el != ref_element_resolved:
            stoi_ratio = stoics[el] / stoics[ref_element_resolved]
            excess = count - stoi_ratio * n_ref
            bulk_ref_en += excess * bulk_energies_resolved.get(el, 0.0)
    surf_en -= bulk_ref_en

    pot_term = 0.0
    for el, count in counts.items():
        if el != ref_element_resolved:
            stoi_ratio = stoics[el] / stoics[ref_element_resolved]
            excess = count - stoi_ratio * n_ref
            pot_term += excess * chem_pots.get(el, 0.0)
    surf_en -= pot_term

    return surf_en


def calculate_surface_energies_agxoy(
    symbol_lists: list[list[str]],
    energies: np.ndarray,
    chem_pots: dict[str, float] | None = None,
) -> np.ndarray:
    """Vectorized AgxOy surface energy: Ω = E − n_Ag·E_Ag_bulk − n_O·(E_O_ref + Δμ_O).

    Ag is the reference element, so its chemical potential is implicitly fixed to bulk
    (n_Ag × E_Ag_bulk). Only chem_pots["O"] enters the result; chem_pots["Ag"] is ignored.

    Args:
        symbol_lists: per-structure chemical-symbol lists (template + adsorbate atoms).
        energies:     MACE total energies, shape (N,).
        chem_pots:    deviation dict, e.g. {"Ag": 0.0, "O": -0.5}.  Defaults to μ=0.
    """
    from dss.data.constants.agxoy import AGXOY_BULK_ENERGIES

    e_ag = AGXOY_BULK_ENERGIES["Ag"]
    e_o = AGXOY_BULK_ENERGIES["O"]
    mu_O = chem_pots.get("O", 0.0) if chem_pots else 0.0
    n_ag = np.array([syms.count("Ag") for syms in symbol_lists])
    n_o = np.array([syms.count("O") for syms in symbol_lists])
    return np.asarray(energies).flatten() - n_ag * e_ag - n_o * (e_o + mu_O)


def calculate_excess(
    atoms: object,
    target_element: str = "Sr",
    ref_element: str = "Ti",
    stoics: dict | None = None,
) -> float:
    """Calculate excess of target element relative to reference element."""
    if stoics is None:
        if ref_element == "Ag":
            from dss.data.constants.agxoy import AGXOY_STOICS
            stoics = AGXOY_STOICS
        else:
            from dss.data.constants.sto import STO_STOICS
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
    figsize: tuple = (2.2, 2.2),
    xlims: tuple[float, float] = (-6, 6),
    stripe_size: float = 100,
    stripe_linewidth: float = 1.5,
    energies: list[float] | None = None,
    offset_data: dict | None = None,
    bulk_energies: dict[str, float] | None = None,
    return_energies: bool = False,
    target_element: str = "Sr",
    ref_element: str = "Ti",
) -> list[plt.Figure] | tuple[list[plt.Figure], list[list[float]]]:
    """Generate surface stability plots (Omega_surf vs Gamma) optimized for journal width."""
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
    gammas = []
    stoics = offset_data["stoics"] if offset_data is not None else None
    for d in data_list:
        g = calculate_excess(d["atoms"], target_element, ref_element, stoics=stoics)
        gammas.append(g)

    for i, pot in enumerate(ref_chem_pots):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(xlims)
        
        model_data = {} # label -> {'gamma': [], 'omega': [], 'color': str}
        ref_points = []

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
            color = d.get("color", "orange")

            if "Sample" in label or "VSSR" in label or "snowyflow" in label or "dss" in label:
                if label not in model_data:
                    model_data[label] = {'gamma': [], 'omega': [], 'color': color}
                model_data[label]['gamma'].append(gammas[j])
                model_data[label]['omega'].append(omega)
            else:
                ref_points.append({
                    "gamma": gammas[j],
                    "omega": omega,
                    "label": label,
                    "marker": d.get("marker", "X"),
                    "color": d.get("color", "black"),
                    "s": d.get("size", 100),
                })

        # Plot Samples as horizontal lines
        for label, m_data in model_data.items():
            ax.scatter(
                m_data['gamma'],
                m_data['omega'],
                c=m_data['color'],
                alpha=0.4,
                marker="_",
                s=stripe_size,
                linewidths=stripe_linewidth,
                label=label,
            )

        # Plot References
        for pt in ref_points:
            ax.scatter(
                [pt["gamma"]], [pt["omega"]],
                label=pt["label"], marker=pt["marker"],
                s=pt["s"], edgecolors="k", c=pt["color"], zorder=10
            )

        # Min line
        all_omegas = [o for md in model_data.values() for o in md['omega']] + [p["omega"] for p in ref_points]
        if all_omegas:
            min_omega = min(all_omegas)
            ax.axhline(
                min_omega, 
                color="#D81B60", 
                linestyle="--", 
                alpha=0.8, 
                linewidth=1,
                label=r"Min. $\Omega_{\text{surf}}$"
            )
            # Add text label for the value
            ax.text(
                np.mean(ax.get_xlim()),
                min_omega,
                f"{min_omega:.1f} eV",
                va="bottom",
                ha="center",
                color="black",
                fontsize=8,
                backgroundcolor="white",
            )

        mu_str = ", ".join([rf"$\mu_{{{k}}} = {v}$" for k, v in pot.items() if v != 0])
        if not mu_str: mu_str = r"$\mu = 0$ eV"

        ax.set_title(mu_str, fontsize=10)
        ax.set_xlabel(rf"$\Gamma_{{\text{{{target_element}}}}}^{{\text{{{ref_element}}}}}$", fontsize=9)
        ax.set_ylabel(r"$\Omega_{\text{surf}}$ [eV]", fontsize=9)
        ax.tick_params(direction="in", labelsize=8)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Legend for every subplot - Boxed
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            by_label = dict(zip(labels, handles, strict=False))
            ax.legend(
                by_label.values(), 
                by_label.keys(), 
                fontsize=8, 
                frameon=True, 
                framealpha=1, 
                edgecolor="black"
            )

        plt.tight_layout()

        if save_dir:
            path = Path(save_dir) / f"surface_stability_{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches="tight")

        figs.append(fig)
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
    """Return for each chem_pot the top k structures with lowest surface energy."""
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
