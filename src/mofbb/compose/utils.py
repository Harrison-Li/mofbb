"""I/O and geometry helpers shared across the PORMAKE pipeline.
https://github.com/Sangwon91/PORMAKE/blob/main/src/pormake/utils.py
This module collects the low-level utilities that translate between the
file formats PORMAKE consumes (``.cgd`` nets, building-block ``.xyz``
files) and the in-memory :class:`ase.Atoms` representation, plus small
geometric helpers used during assembly and a CIF writer for individual
molecules.

It also defines the module-level constant ``METAL_LIKE``: a list of
element symbols treated as "metal-like". It is used elsewhere to detect
which nodes correspond to metal clusters when classifying building
blocks, since metal nodes and organic linkers play different roles in a
Metal-Organic Framework.
"""

from pathlib import Path

import ase
import ase.io
import numpy as np

from mofbb.compose.formats.cgd import read_cgd
from mofbb.compose.log import logger

# Metal species.
METAL_LIKE = [
    "Li",
    "Be",
    "B",
    "Na",
    "Mg",
    "Al",
    "Si",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
]


def wrap_fractional(values, tolerance=1.0e-12):
    """Return canonical fractional coordinates in the interval ``[0, 1)``.

    Floating-point arithmetic can place a coordinate infinitesimally below
    zero, which a plain modulo operation represents as a value just below one.
    Coordinates within ``tolerance`` of either periodic boundary therefore map
    to exactly zero. This keeps equivalent boundary sites in one stable
    representation without moving coordinates away from special positions.

    Parameters
    ----------
    values : array-like
        Fractional coordinates of any shape.
    tolerance : float, optional
        Absolute distance from zero or one treated as the periodic boundary.

    Returns
    -------
    numpy.ndarray
        Wrapped coordinates with numerical boundary noise canonicalized.
    """

    wrapped = np.mod(values, 1.0)
    on_boundary = np.isclose(wrapped, 0.0, atol=tolerance, rtol=0.0) | np.isclose(
        wrapped,
        1.0,
        atol=tolerance,
        rtol=0.0,
    )
    return np.where(on_boundary, 0.0, wrapped)




def read_budiling_block_xyz(bb_file):
    """Parse a building-block ``.xyz`` file into an :class:`ase.Atoms`.

    Reads an extended ``.xyz`` describing a molecular building block:
    each atom line carries an element symbol, Cartesian position, and an
    optional partial charge. Dummy ``"X"`` atoms mark the connection
    points where the fragment attaches to neighboring blocks, and their
    indices are recorded so the assembly step can align them onto a
    topology slot. Any lines beyond the atom block are parsed as bonds
    (atom-index pairs plus a bond type) for CIF export.

    Parameters
    ----------
    bb_file : str or pathlib.Path
        Path to the building-block ``.xyz`` file.

    Returns
    -------
    ase.Atoms
        The building block, whose ``info`` dict holds the connection
        point indices (``"cpi"``), name, and optional ``bonds`` and
        ``bond_types``. Per-atom partial charges are stored as the
        atoms' initial charges.

    Notes
    -----
    The misspelled function name (``budiling``) is intentional and kept
    for backward compatibility.
    """
    name = Path(bb_file).stem

    with open(bb_file, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0])

    symbols = []
    positions = []
    charges = []
    connection_point_indices = []
    for i, line in enumerate(lines[2 : n_atoms + 2]):
        tokens = line.split()
        symbol = tokens[0]
        position = [float(v) for v in tokens[1:4]]
        if len(tokens) == 5:
            charge = float(tokens[4])
        else:
            charge = 0.0

        symbols.append(symbol)
        positions.append(position)
        charges.append(charge)
        if symbol == "X":
            connection_point_indices.append(i)

    bonds = None
    bond_types = None
    if len(lines) > n_atoms + 2:
        logger.debug("There are bonds in building block xyz. Reading...")
        bonds = []
        bond_types = []

        for line in lines[n_atoms + 2 :]:
            tokens = line.split()
            if len(tokens) < 3:
                logger.debug("%s: len(line.split()) < 3 %s", bb_file, line)
                continue

            i = int(tokens[0])
            j = int(tokens[1])
            t = tokens[2]
            bonds.append((i, j))
            bond_types.append(t)
        bonds = np.array(bonds)

    info = {}
    info["cpi"] = connection_point_indices
    info["name"] = name
    info["bonds"] = bonds
    info["bond_types"] = bond_types

    atoms = ase.Atoms(symbols=symbols, positions=positions, charges=charges, info=info)

    return atoms


def write_molecule_cif(filename, atoms, bond_pairs, bond_types):
    """Write a single molecule to a P1 CIF file with bonds.

    Places the molecule, centered on its center of mass, inside a cubic
    ``P1`` cell sized to comfortably enclose it, then writes fractional
    coordinates, partial charges, and the supplied bond list. This is
    used to export an individual building block or fragment for
    inspection in crystallographic viewers.

    Parameters
    ----------
    filename : str or pathlib.Path
        Output path; a ``.cif`` suffix is enforced.
    atoms : ase.Atoms
        The molecule to write.
    bond_pairs : list of tuple of int
        Bonded atom-index pairs ``(i, j)``.
    bond_types : list of str
        Bond type for each pair, one of ``"S"``, ``"D"``, ``"T"``, or
        ``"A"`` (single, double, triple, aromatic).

    Returns
    -------
    None
    """

    path = Path(filename).resolve()
    if path.suffix != ".cif":
        path = path.with_suffix(".cif")

    stem = path.stem.replace(" ", "_")
    with path.open("w") as f:
        f.write("data_{}\n".format(stem))

        f.write("_symmetry_space_group_name_H-M    P1\n")
        f.write("_symmetry_Int_Tables_number       1\n")
        f.write("_symmetry_cell_setting            triclinic\n")

        f.write("loop_\n")
        f.write("_symmetry_equiv_pos_as_xyz\n")
        f.write("'x, y, z'\n")

        # Calculate cell parameters.
        positions = atoms.get_positions()
        com = atoms.get_center_of_mass()

        distances = np.linalg.norm(positions - com, axis=1)
        max_distances = np.max(distances)

        box_length = 2 * max_distances + 4

        f.write("_cell_length_a     {:.3f}\n".format(box_length))
        f.write("_cell_length_b     {:.3f}\n".format(box_length))
        f.write("_cell_length_c     {:.3f}\n".format(box_length))
        f.write("_cell_angle_alpha  90.0\n")
        f.write("_cell_angle_beta   90.0\n")
        f.write("_cell_angle_gamma  90.0\n")

        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        f.write("_atom_type_partial_charge\n")

        # Get fractional coordinates
        # fractional coordinate of C.O.M is (0.5, 0.5, 0.5).
        symbols = atoms.get_chemical_symbols()
        charges = atoms.get_initial_charges()
        fracts = (positions - com) / box_length + 0.5

        # Write label and pos information.
        for i, (sym, fract, charge) in enumerate(zip(symbols, fracts, charges)):
            label = "{}{}".format(sym, i)
            f.write(
                "{} {} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(label, sym, *fract, charge)
            )

        # Write bonds information.
        f.write("loop_\n")
        f.write("_geom_bond_atom_site_label_1\n")
        f.write("_geom_bond_atom_site_label_2\n")
        f.write("_geom_bond_distance\n")
        f.write("_geom_bond_site_symmetry_2\n")
        f.write("_ccdc_geom_bond_type\n")

        for (i, j), t in zip(bond_pairs, bond_types):
            label_i = "{}{}".format(symbols[i], i)
            label_j = "{}{}".format(symbols[j], j)

            distance = np.linalg.norm(positions[i] - positions[j])

            f.write("{} {} {:.3f} . {}\n".format(label_i, label_j, distance, t))
