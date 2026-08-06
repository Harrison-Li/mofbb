"""Read conventional and decorated CGD topology files."""

from __future__ import annotations

from pathlib import Path

import ase
import ase.neighborlist
from ase.geometry import cellpar_to_cell
import numpy as np
import pymatgen.core as mg

from mofbb.compose.log import logger
from mofbb.compose.formats.meta import CgdMeta
from mofbb.compose.formats.records import CgdData


class CgdReader:
    """Parse one conventional or explicitly expanded CGD topology."""

    def __init__(
        self,
        path: str | Path,
        node: str = "C",
        edge: str = "O",
    ) -> None:
        self.path = Path(path)
        self.node = node
        self.edge = edge

    def read(self) -> ase.Atoms:
        with self.path.open("r", encoding="utf-8") as handle:
            # Neglect "CRYSTAL" and "END"
            raw = handle.readlines()[1:-1]

        meta = CgdMeta.from_lines(raw)
        lines = [line for line in raw if not line.lstrip().startswith("#")]
        data = CgdData.from_lines(lines)
        props = data.props(meta)
        maps = meta.slots(data.num_nodes + data.num_edges)
        meta.check_ports(props["cn"])

        group = data.spacegroup.replace(" ", "")
        if group == "P1" or meta.expanded:
            return self._explicit_atoms(data, meta, props, maps)
        return self._expanded_atoms(data, props)

    def _explicit_atoms(
        self,
        data: CgdData,
        meta: CgdMeta,
        props: dict[str, list[int]],
        maps: list[list[int]] | None,
    ) -> ase.Atoms:
        info = {
            "spacegroup": data.spacegroup,
            "name": data.name,
            "cn": props["cn"],
        }
        meta.add_info(info, maps)
        if len(data.edge_ends) == data.num_edges:
            info["edge_endpoints"] = data.edge_ends
        return ase.Atoms(
            symbols=data.species(self.node, self.edge),
            scaled_positions=np.mod(data.coords(), 1.0),
            cell=cellpar_to_cell(data.cell),
            tags=props["type"],
            pbc=True,
            info=info,
        )

    def _expanded_atoms(
        self,
        data: CgdData,
        props: dict[str, list[int]],
    ) -> ase.Atoms:
        group = data.spacegroup
        # I don't know why pymatgen can't parse this spacegroup.
        if group == "Cmca":
            group = "Cmce"

        structure = mg.Structure.from_spacegroup(
            sg=group,
            lattice=mg.Lattice.from_parameters(*data.cell),
            species=data.species(self.node, self.edge),
            coords=data.coords(),
            site_properties=props,
        )

        # Add information.
        info = {
            "spacegroup": group,
            "name": data.name,
            "cn": structure.site_properties["cn"],
        }

        # Cast mg.Structure to ase.Atoms
        atoms = ase.Atoms(
            symbols=[species.name for species in structure.species],
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            tags=structure.site_properties["type"],
            pbc=True,
            info=info,
        )
        self._remove_overlapping_copies(atoms)
        return atoms

    @staticmethod
    def _remove_overlapping_copies(atoms: ase.Atoms) -> None:
        # Remove overlap
        first, second = ase.neighborlist.neighbor_list(
            "ij",
            atoms,
            cutoff=0.1,
        )
        tags = atoms.get_tags()
        removed = {
            int(j)
            for i, j in zip(first, second)
            if j > i and tags[i] == tags[j]
        }
        if not removed:
            return

        # Save original size of atoms.
        size = len(atoms)
        del atoms[list(removed)]

        cn = atoms.info["cn"]
        # Remove old cn info.
        atoms.info["cn"] = [cn[i] for i in range(size) if i not in removed]
        logger.debug("Overlapped symmetry copies are removed: %s", removed)


def read_cgd(filename, node_symbol="C", edge_center_symbol="O"):
    """Parse a ``.cgd`` net file into an :class:`ase.Atoms` topology.

    A ``.cgd`` file describes an abstract net by its space group, cell
    parameters, symmetry-unique node positions (with coordination
    numbers), and edges. This function expands the asymmetric unit by
    the space group via :mod:`pymatgen`, places a placeholder atom at
    every node and edge center, removes symmetry-generated duplicates,
    and returns the result as an :class:`ase.Atoms` object that the rest
    of PORMAKE treats as the topology skeleton.

    The one-based identifier in each ``NODE`` record becomes its zero-based
    non-negative type tag, so repeated identifiers share one building-block
    type. Edge centers use negative tags, and every site carries its
    coordination number ``cn`` for connectivity matching. Decomposer-written
    ``# EXPANDED`` files retain their explicit sites and embedded operations.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the ``.cgd`` net file.
    node_symbol : str, optional
        Element symbol used as a placeholder for net nodes.
    edge_center_symbol : str, optional
        Element symbol used as a placeholder for edge centers.

    Returns
    -------
    ase.Atoms
        The topology, with ``tags`` encoding site types and ``info``
        holding the space group, name, and per-site coordination
        numbers.
    """
    return CgdReader(filename, node_symbol, edge_center_symbol).read()
