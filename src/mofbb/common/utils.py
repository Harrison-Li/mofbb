'''
Coordinate helpers and SMILES-to-XYZ conversion.
'''

import os
from collections.abc import Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def minimum_image_delta(
    cell: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    '''
    Wrap a displacement into the central fractional cell.

    Fractional components are rounded independently. For a skewed cell, this
    need not be the shortest Cartesian image vector.

    Args:
        cell (numpy.ndarray):
            (3, 3) cell matrix with lattice vectors as rows.
        start (numpy.ndarray):
            (3,) Cartesian starting position.
        end (numpy.ndarray):
            (3,) Cartesian ending position in the same units as cell.

    Returns:
        A (3,) Cartesian displacement. For a singular cell, return end - start.
    '''

    cell = np.asarray(cell, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if abs(np.linalg.det(cell)) < 1.0e-12:
        return end - start
    frac_delta = np.linalg.solve(cell.T, end - start)
    frac_delta -= np.round(frac_delta)
    return frac_delta @ cell


def cart_to_frac(cell: np.ndarray, cart: np.ndarray) -> np.ndarray:
    '''
    Convert a Cartesian vector to fractional coordinates.

    Args:
        cell (numpy.ndarray):
            (3, 3) cell matrix with lattice vectors as rows.
        cart (numpy.ndarray):
            (3,) Cartesian vector in the same units as cell.

    Returns:
        A (3,) fractional vector, or a copy of cart when the cell is singular.
    '''

    cell = np.asarray(cell, dtype=float)
    cart = np.asarray(cart, dtype=float)
    if abs(np.linalg.det(cell)) < 1.0e-12:
        return np.array(cart, dtype=float)
    return np.linalg.solve(cell.T, cart)


def unwrap_fractional_near(
    frac_position: Iterable[float], reference_frac: Iterable[float]
) -> tuple[float, float, float]:
    '''
    Shift a fractional position to the central image around a reference.

    Args:
        frac_position (iterable of float):
            Three fractional coordinates to shift.
        reference_frac (iterable of float):
            Three fractional coordinates defining the reference image.

    Returns:
        A three-float tuple whose displacement from the reference lies within
        half a cell along each fractional axis.
    '''

    frac = np.array(tuple(frac_position), dtype=float)
    reference = np.array(tuple(reference_frac), dtype=float)
    return tuple(float(value) for value in frac - np.round(frac - reference))


def center_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    '''
    Translate positions so their centroid lies at the unit-cell center.

    Args:
        positions (numpy.ndarray):
            (n, 3) Cartesian positions.
        cell (numpy.ndarray):
            (3, 3) cell matrix in the same units, with lattice vectors as rows.

    Returns:
        An (n, 3) translated array. Empty input is returned unchanged.
    '''

    if len(positions) == 0:
        return positions
    centroid = positions.mean(axis=0)
    cell_center = np.sum(cell, axis=0) * 0.5
    return positions - centroid + cell_center


def wrap_frac(values, tolerance=1.0e-12):
    '''
    Wrap fractional coordinates into [0, 1), snapping cell boundaries to zero.

    Args:
        values (array-like):
            Fractional coordinates of any shape.
        tolerance (float):
            Absolute tolerance for treating a wrapped value as a cell boundary.

    Returns:
        A NumPy array with the same shape as values.
    '''

    wrapped = np.mod(values, 1.0)
    on_boundary = np.isclose(wrapped, 0.0, atol=tolerance, rtol=0.0) | np.isclose(
        wrapped,
        1.0,
        atol=tolerance,
        rtol=0.0,
    )
    return np.where(on_boundary, 0.0, wrapped)


def smiles_to_xyz(smiles, out_xyz, random_seed=42):
    '''
    Build a hydrogen-complete 3D molecule from SMILES and save an XYZ file.

    Embedding uses ETKDG, followed by UFF optimization with an MMFF fallback.
    If both optimizers fail, the embedded coordinates are still written.

    Args:
        smiles (str):
            Molecule in SMILES notation.
        out_xyz (str or path-like):
            Output XYZ path. Parent directories are created if needed.
        random_seed (int):
            Seed for the first embedding attempt; the fallback uses RDKit defaults.

    Returns:
        The supplied output path. Invalid SMILES or failed embedding raises an error.
    '''
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
