import numpy as np
import ase.io
from ase import Atoms, units
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE, LBFGS


def optimize_geometry(
    atoms: Atoms,
    charge: int,
    multiplicity: int,
    config: dict,
    outputfile: str = "mof_opt.cif",
) -> dict:
    """Optimize the atomic positions and unit cell of a MOF.
    """
    if config is None:
        config = {}

    atoms.info["charge"] = charge
    atoms.info["spin"] = multiplicity
    atoms.pbc = config.get("pbc", True)

    calculator = config.get("calculator")
    mlip: str = config.get("mlip", "dpa")
    if calculator is not None:
        atoms.calc = calculator
    elif mlip.lower() == "dpa":
        from deepmd.calculator import DP
        dpa_calc = DP(model=config["model"])
        atoms.calc = dpa_calc
    elif mlip.lower() == "mace":
        from mace.calculators import mace_omat
        mace_calc = mace_omat(
            model=config["model"],
            device=device,
            #default_dtype=precision,
        )
        atoms.calc = mace_calc
    else:
        raise ValueError(f"Unsupported MLIP model: {mlip}")

    pressure_gpa = float(config.get("pressure_GPa", config.get("pressure", 0.0)))
    if config.get("relax_cell", True):
        opt_atoms = FrechetCellFilter(
            atoms,
            scalar_pressure=pressure_gpa * units.GPa,
        )
    else:
        opt_atoms = atoms
        print("Cell is fixed, only coordinates will be optimized")

    optimizer_name: str = config.get("optimizer", "LBFGS")
    maxstep = config.get("maxstep", None)
    logfile = config.get("logfile", "-")
    if optimizer_name.lower() == "bfgs":
        optimizer = BFGS(opt_atoms, maxstep=maxstep, logfile=logfile)
    elif optimizer_name.lower() == "fire":
        optimizer = FIRE(opt_atoms, maxstep=maxstep, logfile=logfile)
    elif optimizer_name.lower() == "lbfgs":
        optimizer = LBFGS(opt_atoms, maxstep=maxstep, logfile=logfile)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    fmax = float(config.get("fmax", 0.05))
    max_steps: int = config.get("max_steps", 1000)
    converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
    if not converged:
        print("Optimization did not converge within the maximum number of steps.")

    ase.io.write(outputfile, atoms, format="cif")
    print(f"Optimized MOF structure saved to {outputfile}")

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    atomic_fmax = float(np.linalg.norm(forces, axis=1).max())

    # This is the force array ASE's optimizer actually uses for convergence.
    # With FrechetCellFilter it contains one row per atom followed by three
    # generalized cell-force rows.  Reporting only the raw atomic forces can
    # therefore make a correctly non-converged cell relaxation look converged.
    optimization_forces = np.asarray(opt_atoms.get_forces(), dtype=float)
    filter_fmax = float(np.linalg.norm(optimization_forces, axis=1).max())
    if isinstance(opt_atoms, FrechetCellFilter):
        cell_forces = optimization_forces[len(atoms) :]
        cell_filter_fmax = float(np.linalg.norm(cell_forces, axis=1).max())
    else:
        cell_filter_fmax = 0.0

    return {
        "energy": energy,
        "forces": forces,
        # This is the same maximum generalized-force norm tested by
        # optimizer.run(fmax=...).  Keep the component values separately.
        "fmax": filter_fmax,
        "atomic_fmax": atomic_fmax,
        "filter_fmax": filter_fmax,
        "cell_filter_fmax": cell_filter_fmax,
        "stress": stress,
        "converged": converged,
        "n_steps": optimizer.nsteps,
    }
