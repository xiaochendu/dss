import os
import sys
import argparse
from pathlib import Path
import torch
import ase.io
import numpy as np

# Add dss to path dynamically based on script location
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from dss.helpers import get_diffusion_model, sample
from dss.data.constants.agxoy import number_to_element, mask_index
from dss.energy.mace import MACEEnergyModel

def main():
    parser = argparse.ArgumentParser(description="Grid sampling for DSS AgxOy.")
    parser.add_argument("--checkpoint", required=True, help="Path to the dss model checkpoint (.ckpt)")
    parser.add_argument("--template", required=True, help="Path to the template .xyz file")
    parser.add_argument("--out_dir", required=True, help="Directory to save the sampled xyz files")
    parser.add_argument("--ag_min", type=int, default=4, help="Minimum Ag count")
    parser.add_argument("--ag_max", type=int, default=6, help="Maximum Ag count")
    parser.add_argument("--o_min", type=int, default=2, help="Minimum O count")
    parser.add_argument("--o_max", type=int, default=5, help="Maximum O count")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples per composition")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument("--n_atom_basis", type=int, default=32, help="n_atom_basis for the score model")
    parser.add_argument("--n_interactions", type=int, default=4, help="n_interactions for the score model")
    parser.add_argument("--n_rbf", type=int, default=30, help="n_rbf for the score model")
    parser.add_argument("--z_conf_min", type=float, default=2.8613568, help="z_confinement minimum")
    parser.add_argument("--z_conf_max", type=float, default=6.49999996, help="z_confinement maximum")
    
    args = parser.parse_args()

    ag_range = range(args.ag_min, args.ag_max + 1)
    o_range = range(args.o_min, args.o_max + 1)
    z_confinement = [args.z_conf_min, args.z_conf_max]

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    template_atoms = ase.io.read(args.template)
    
    print(f"Loading DSS model from {args.checkpoint}...")
    mace_energy_model = MACEEnergyModel(
        model="medium",
        device=str(device),
        default_dtype="float32",
        number_to_element={k: v for k, v in number_to_element.items() if k <= mask_index}
    )
    
    diffusion, _ = get_diffusion_model(
        n_atom_basis=args.n_atom_basis,
        n_interactions=args.n_interactions,
        n_rbf=args.n_rbf,
        potential_model_instance=mace_energy_model,
    )
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    diffusion.load_state_dict(ckpt["state_dict"])
    diffusion.to(device)
    diffusion.eval()
    
    print(f"Starting DSS Grid Sampling in {args.out_dir}")

    for ag in ag_range:
        for o in o_range:
            output_dir = os.path.join(args.out_dir, f"Ag{ag}O{o}")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "sample.xyz")
            
            symbols = ["Ag"] * ag + ["O"] * o
            
            print(f"--> Sampling Ag{ag}O{o} (n={args.num_samples})...")
            
            with torch.no_grad():
                sampled_atoms = sample(
                    diffusion=diffusion,
                    num_samples=args.num_samples,
                    template=template_atoms,
                    symbols=symbols,
                    z_confinement=z_confinement,
                    num_steps=args.num_steps,
                    return_trajectories=False
                )
            
            ase.io.write(output_file, sampled_atoms, format="extxyz")
            print(f"    Done. Saved to {output_file}")

    print("\nDSS Grid Sampling Complete.")

if __name__ == "__main__":
    main()
