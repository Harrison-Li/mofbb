"""Relax periodic MOF structures from a YAML configuration."""

import argparse
from pathlib import Path

from mofbb.io.structure import read_atoms
import yaml

from mofbb.optimize.mlip_utils import optimize_geometry


def main():
    parser = argparse.ArgumentParser(
        description="Relax periodic MOF structures with a DPA model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mof.yaml"),
        help="path to the YAML configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    opt_config = config["optimization"]
    mof_path = Path(data_config["mof_path"])
    output_dir = Path(config.get("output_dir", "mof_relaxed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if mof_path.is_dir():
        mof_files = sorted(
            path for path in mof_path.iterdir() if path.suffix.lower() == ".cif"
        )
    else:
        mof_files = [mof_path]

    charge = int(data_config.get("charge", 0))
    multiplicity = int(data_config.get("multiplicity", 1))

    for mof_file in mof_files:
        output_file = output_dir / f"{mof_file.stem}_opt.cif"
        print(f"Relaxing MOF: {mof_file}")
        atoms = read_atoms(mof_file)
        opt_info = optimize_geometry(
            atoms=atoms,
            charge=charge,
            multiplicity=multiplicity,
            config=opt_config,
            outputfile=str(output_file),
        )
        print(
            f"Final energy: {opt_info['energy']:.6f} eV; "
            f"fmax: {opt_info['fmax']:.6f} eV/Ang; "
            f"steps: {opt_info['n_steps']}; "
            f"converged: {opt_info['converged']}"
        )


if __name__ == "__main__":
    main()
