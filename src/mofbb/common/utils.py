import ase
import numpy as np
import os

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

def wrap_frac(values, tolerance=1.0e-12):
    """Return canonical fractional coordinates in the interval ``[0, 1)``."""

    wrapped = np.mod(values, 1.0)
    on_boundary = np.isclose(wrapped, 0.0, atol=tolerance, rtol=0.0) | np.isclose(
        wrapped,
        1.0,
        atol=tolerance,
        rtol=0.0,
    )
    return np.where(on_boundary, 0.0, wrapped)


def read_bb_xyz(bb_file):
    """Parse a building-block ``.xyz`` file into an :class:`ase.Atoms`."""
    path = Path(bb_file)
    lines = path.read_text(encoding="utf-8").splitlines()
    n_atoms = int(lines[0])
    atoms = [line.split() for line in lines[2 : n_atoms + 2]]
    symbols = [record[0] for record in atoms]
    bond_records = [
        record
        for line in lines[n_atoms + 2 :]
        if len(record := line.split()) >= 3
    ]
    has_bonds = len(lines) > n_atoms + 2
    return ase.Atoms(
        symbols=symbols,
        positions=[[float(value) for value in record[1:4]] for record in atoms],
        charges=[float(record[4]) if len(record) == 5 else 0 for record in atoms],
        info={
            "port_idx": [index for index, symbol in enumerate(symbols) if symbol == "X"],
            "name": path.stem,
            "bonds": (
                np.asarray([[int(record[0]), int(record[1])] for record in bond_records])
                if has_bonds
                else None
            ),
            "bond_types": (
                [record[2] for record in bond_records] if has_bonds else None
            ),
        },
    )



def smiles_to_xyz(smiles, out_xyz, random_seed=42):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()  # Use ETKDG for realistic initial geometry
    params.randomSeed = random_seed
    res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        # try without specifying seed if initial embed fails
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res != 0:
            raise RuntimeError("3D embedding failed")

    # UFF optimization (falls back to MMFF if UFF unavailable)
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as e:
            print("UFF failed:", e)

    conf = mol.GetConformer()
    natoms = mol.GetNumAtoms()

    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(out_xyz)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Write XYZ
    with open(out_xyz, "w") as f:
        f.write(f"{natoms}\n")
        f.write(f"Generated from SMILES: {smiles}\n")
        for i in range(natoms):
            a = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            line = f"{a.GetSymbol():2s} {pos.x: .6f} {pos.y: .6f} {pos.z: .6f}"
            if i < natoms - 1:
                f.write(line + "\n")
            else:
                f.write(line)

    print(f"Wrote {out_xyz} ({natoms} atoms)")
    return out_xyz


def read_atoms(input_path: str):
    """Read input mol.xyz and ASE Atoms object."""
    if os.path.exists(input_path):
        atoms = ase.io.read(input_path)
        # ase.io.read returns a list for multi-frame files, single Atoms otherwise
        if isinstance(atoms, list):
            atoms = atoms[0]
        return atoms
    else:
        raise FileNotFoundError(f"Input file {input_path} does not exist.")