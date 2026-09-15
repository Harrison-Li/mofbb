"""Identify whole organic linkers independently of assembly building blocks."""

from collections import defaultdict
import hashlib
import json

from ase.data import covalent_radii
from ase.neighborlist import neighbor_list
import igraph as ig
import networkx as nx
import numpy as np
from pymatgen.core import Element

from mofbb.compose.connectivity import bond_arrays, minimum_image_shifts
from mofbb.decompose.pbc import PeriodicBond, analyze_component


SINGLE_LINKER_POLICY_VERSION = 1


def _molecule_key(atoms, graph):
    """Canonical element-labeled connectivity, independent of atom ordering."""
    indices = sorted(graph.nodes)
    local = {atom: index for index, atom in enumerate(indices)}
    molecule = ig.Graph(
        n=len(indices),
        edges=[(local[i], local[j]) for i, j in graph.edges],
        vertex_attrs={"number": atoms.numbers[indices].tolist()},
    )
    canonical = molecule.permute_vertices(
        molecule.canonical_permutation(color=molecule.vs["number"])
    )
    return (
        tuple(canonical.vs["number"]),
        tuple(sorted(tuple(sorted(edge)) for edge in canonical.get_edgelist())),
    )


def analyze_linker_types(atoms, bonds=None, bond_shifts=None, *, bond_scale=1.2):
    """Count complete carbon-containing molecules bridging metal atom images.

    Cut metal/nonmetal bonds in the assembled periodic graph, not the node/edge
    boundaries used by the builder. A linker must be a finite nonmetal component
    contacting at least two distinct (metal atom, cell image) pairs. Thus parts
    retained inside metal building blocks participate in the check. Free guests,
    ligands contacting only one metal image, and carbon-free components do not
    count. Coordination to multiple metals within an SBU is counted conservatively.

    Identity uses the full element-labeled molecular graph, including hydrogen;
    it ignores conformation, binding mode, bond order, and stereochemistry. Bonds
    are inferred from current coordinates by default, using the same covalent
    cutoff as the generator's connectivity screen, without trusting CIF tables.
    Explicit bonds, if supplied, must describe the current structure.
    """
    if not np.isfinite(bond_scale) or bond_scale <= 0:
        raise ValueError("Linker bond scale must be positive and finite.")
    if (
        not len(atoms)
        or not np.isfinite(atoms.positions).all()
        or not np.isfinite(atoms.cell).all()
        or (np.any(atoms.pbc) and atoms.cell.volume <= 0)
        or np.any(atoms.numbers == 0)
    ):
        raise ValueError("Linker analysis requires finite geometry without X sites.")
    metals = np.array([Element(symbol).is_metal for symbol in atoms.symbols])
    if not metals.any():
        raise ValueError("Single-linker MOF requires metal atoms.")
    if bonds is None:
        if bond_shifts is not None:
            raise ValueError("Bond shifts require explicit bonds.")
        i, j, shifts = neighbor_list(
            "ijS", atoms, bond_scale * covalent_radii[atoms.numbers]
        )
        keep = (i <= j) & ~(metals[i] & metals[j])
        bonds, bond_shifts = np.column_stack((i[keep], j[keep])), shifts[keep]
    elif bond_shifts is None:
        bond_shifts = minimum_image_shifts(atoms, bonds)
    bonds, shifts = bond_arrays(len(atoms), bonds, bond_shifts)

    organic = nx.Graph()
    organic.add_nodes_from(np.flatnonzero(~metals).tolist())
    records = []
    metal_neighbors = defaultdict(list)
    for (i, j), shift in zip(bonds, shifts, strict=True):
        i, j = int(i), int(j)
        image = tuple(map(int, shift))
        if not metals[i] and not metals[j]:
            organic.add_edge(i, j)
            records.append(PeriodicBond(i, j, image))
        elif metals[i] != metals[j]:
            if metals[i]:
                metal_neighbors[j].append((i, tuple(-value for value in image)))
            else:
                metal_neighbors[i].append((j, image))

    records = tuple(records)
    types = {}
    ignored = 0
    for component in nx.connected_components(organic):
        indices = sorted(component)
        if 6 not in atoms.numbers[indices]:
            continue
        frame = analyze_component(component, records)
        if frame.rank:
            raise ValueError(
                "Single-linker MOF requires finite organic molecules; "
                f"found a {frame.rank}D carbon-containing component."
            )
        contacts = {
            (metal, tuple(a + b for a, b in zip(frame.offset(atom), shift)))
            for atom in component
            for metal, shift in metal_neighbors[atom]
        }
        if len(contacts) < 2:
            ignored += 1
            continue
        key = _molecule_key(atoms, organic.subgraph(component))
        if key not in types:
            types[key] = {
                "formula": atoms[indices].get_chemical_formula(),
                "graph_sha256": hashlib.sha256(json.dumps(key).encode()).hexdigest(),
                "count": 0,
                "num_atoms": len(indices),
            }
        types[key]["count"] += 1

    return {
        "single_linker_policy_version": SINGLE_LINKER_POLICY_VERSION,
        "linker_bond_scale": float(bond_scale),
        "num_linker_types": len(types),
        "num_linkers": sum(item["count"] for item in types.values()),
        "linker_types": sorted(
            types.values(), key=lambda item: (item["formula"], item["graph_sha256"])
        ),
        "ignored_nonbridging_organic_components": ignored,
    }


def validate_single_linker(atoms, bonds=None, bond_shifts=None, *, bond_scale=1.2):
    """Require exactly one molecular linker type in the complete framework."""
    quality = analyze_linker_types(atoms, bonds, bond_shifts, bond_scale=bond_scale)
    if quality["num_linker_types"] != 1:
        detail = ", ".join(
            f"{item['formula']} ({item['count']} copies)"
            for item in quality["linker_types"]
        ) or "no bridging organic molecules"
        raise ValueError(
            "Single-linker MOF required; found "
            f"{quality['num_linker_types']} linker types: {detail}."
        )
    return quality
