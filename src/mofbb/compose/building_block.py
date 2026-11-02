'''
Store building-block atoms, bonds, and dummy X connection sites.
'''
import copy
from dataclasses import dataclass
from functools import cached_property

import networkx as nx
import numpy as np
from ase import Atoms
from pymatgen.core import Molecule
from pymatgen.core.structure import FunctionalGroups

from ase.data import covalent_radii
from ase.neighborlist import natural_cutoffs

from mofbb.common.chemistry import coordination_profile
from mofbb.common.geometry import normalize_directions
from mofbb.io.xyz import read_bb_xyz
from mofbb.decompose.constants import is_metal


@dataclass(frozen=True)
class EdgeFrame:
    '''
    Store an edge frame and the ambiguity of its transverse orientation.

    Args:
        axes (numpy.ndarray):
            A (3, 3) array whose rows are the unit x, y, z axes.
        pca_gap (float):
            Relative gap between the two largest transverse PCA eigenvalues.
        torsion_degenerate (bool):
            Whether the atomic geometry leaves the torsion poorly defined.
        sign_degenerate (bool):
            Whether the atomic geometry leaves the transverse axis sign ambiguous.
    '''

    axes: np.ndarray
    pca_gap: float
    torsion_degenerate: bool
    sign_degenerate: bool

    @property
    def x(self):
        '''
        Read the local x axis.

        Returns:
            A (3,) unit vector from row 0 of axes.
        '''
        return self.axes[0]

    @property
    def y(self):
        '''
        Read the local y axis.

        Returns:
            A (3,) unit vector from row 1 of axes.
        '''
        return self.axes[1]

    @property
    def z(self):
        '''
        Read the local z axis.

        Returns:
            A (3,) unit vector from row 2 of axes.
        '''
        return self.axes[2]


def _unit_vector(vector, name, frame_epsilon=1e-12):
    """Return ``vector`` normalized to unit length."""

    vector = np.asarray(vector, frame_epsilon, dtype=float)
    length = np.linalg.norm(vector)
    if length < frame_epsilon:
        raise ValueError(f"Cannot define {name} from a near-zero vector.")
    return vector / length


def _orient_pca_axis(axis, transverse, axial, pca_tol=1e-8, frame_epsilon=1e-12):
    """Choose the sign of a PCA axis from the full atomic distribution."""

    projection = transverse @ axis
    scale = np.max(np.abs(projection))
    if scale < frame_epsilon:
        return axis, True

    p = projection / scale
    axial_scale = np.max(np.abs(axial))
    if axial_scale < frame_epsilon:
        axial = np.zeros_like(axial)
    else:
        axial = axial / axial_scale

    scores = (
        np.mean(p * axial),
        np.mean(p**3),
        np.mean(p * axial**2),
        np.mean(p**3 * axial),
        np.mean(p**5),
    )
    for score in scores:
        if abs(score) >= pca_tol:
            return (axis if score > 0.0 else -axis), False

    for projection in p:
        if abs(projection) >= pca_tol:
            return (axis if projection > 0.0 else -axis), True
    return axis, True


class BuildingBlock:
    '''
    Store a molecular fragment and its X connection sites.

    Args:
        atoms (ase.Atoms):
            Atoms with building-block metadata; use read(path) to load a file.
    '''

    def __init__(self, atoms):
        '''
        Initialize a building block from atoms, ports, and stored bonds.

        Args:
            atoms (ase.Atoms):
                Atoms with building-block metadata.
        '''
        if not isinstance(atoms, Atoms):
            raise TypeError("Use .read(path) to load a building-block file.")
        self.atoms = atoms
        self.name = self.atoms.info["name"]
        self.port_idx = np.array(self.atoms.info["port_idx"])
        self._bonds = self.atoms.info["bonds"]
        self._bond_types = self.atoms.info["bond_types"]

    @classmethod
    def read(cls, bb_file):
        """Read building-block atoms and metadata from an XYZ file.

        Args:
            bb_file (str or Path):
                Building-block XYZ file, optionally followed by bond records.

        Returns:
            A new instance of the requested class with atoms, ports, and bonds.
        """
        return cls(read_bb_xyz(bb_file))

    def copy(self):
        '''
        Copy the block so its atoms, ports, and bonds can be edited separately.

        Returns:
            An independent BuildingBlock of the same class as this block.
        '''
        block = copy.copy(self)
        block.atoms = self.atoms.copy()
        block.port_idx = self.port_idx.copy()
        block._bonds = self.bonds.copy()
        block._bond_types = self.bond_types.copy()
        return block

    def validate_connectivity(self, bond_scale=1.2):
        '''
        Check that the real atoms form one connected, chemically valid fragment.

        Raises ValueError if the real-atom graph is empty or disconnected, or
        if a known coordination constraint is violated.

        Args:
            bond_scale (float):
                Multiplier for summed covalent radii when checking stored bond lengths.
        '''
        bonds = np.asarray(self.bonds, dtype=int).reshape(-1, 2)
        real = self.atoms.numbers != 0
        bonds = bonds[real[bonds].all(axis=1)]
        distances = np.linalg.norm(
            self.atoms.positions[bonds[:, 1]] - self.atoms.positions[bonds[:, 0]], axis=1
        )
        limits = bond_scale * covalent_radii[self.atoms.numbers[bonds]].sum(axis=1)
        graph = nx.Graph()
        graph.add_nodes_from(np.flatnonzero(real))
        graph.add_edges_from(bonds[(distances > 1e-8) & (distances <= limits)])
        if not graph or not nx.is_connected(graph):
            raise ValueError(
                f"Building block {self.name!r} must have one connected real-atom component."
            )
        self.validate_chemistry()

    def validate_chemistry(self):
        '''
        Check the known neighbors of each atom before joining blocks.

        X partner identities are checked during assembly. Stored charges are
        ignored, and invalid coordination raises ValueError.

        Returns:
            The coordination profile, including nonmetal neighbor counts.
        '''
        profile = coordination_profile(self.atoms.numbers, self.bonds)
        if profile.error is not None:
            raise ValueError(f"{self.name!r}: {profile.error}")
        return profile

    def connection_directions(self) -> np.ndarray:
        '''
        Calculate unit vectors from the X-port centroid to each port.

        Returns:
            An (n_ports, 3) array of unit vectors in port_idx order.
        '''
        return normalize_directions(self.port_pos)

    def set_centroid(self, centroid):
        '''
        Translate all atoms so the X-port centroid reaches the given point.

        Args:
            centroid (array-like):
                Target Cartesian position with shape (3,), in angstroms.
        '''
        self.atoms.set_positions(self.atoms.positions - self.centroid + centroid)

    def translate(self, displacement):
        self.atoms.translate(displacement)
        return self

    def rotate(self, angle, axis="z", center=None):
        '''
        Rotate all atoms, including X ports, around a fixed point.

        Args:
            angle (float):
                Rotation angle in degrees, following the right-hand rule.
            axis (str or array-like):
                Axis such as "x", "y", "z", or a three-component vector.
            center (array-like or None):
                Fixed Cartesian point in angstroms. Defaults to the X-port
                centroid. The unit cell is unchanged.

        Returns:
            This building block after rotation. Use copy() first to keep
            the original coordinates.
        '''
        self.atoms.rotate(
            angle, axis, center=self.centroid if center is None else center,
        )
        return self

    def functionalize(self, index: int, functional_group: str):
        '''
        Replace a terminal atom with a named pymatgen functional group.

        X ports retain their positions and order. Pymatgen places the group;
        no geometry relaxation or clash search is performed.

        Args:
            index (int):
                Zero-based index of the terminal real atom to replace.
            functional_group (str):
                Name in pymatgen.core.structure.FunctionalGroups, such as
                "amine", "hydroxyl", "fluoro", or "methyl".

        Returns:
            This building block after substitution. Surviving host atoms
            retain their order, followed by the group atoms. Existing bond
            labels are kept; new bonds are recorded as single bonds, as in
            calc_bonds_dist().
        '''
        if index in self.port_idx:
            raise ValueError("Cannot replace an X connection point.")

        bonds = np.asarray(self.bonds, dtype=int).reshape(-1, 2)
        removed = np.any(bonds == index, axis=1)
        anchor, = bonds[removed][bonds[removed] != index]
        symbols = self.atoms.get_chemical_symbols()
        molecule = Molecule(
            ["H" if s == "X" else s for s in symbols], self.atoms.positions,
        )
        molecule.substitute(index, FunctionalGroups[functional_group].copy())

        keep = np.arange(self.n_atoms) != index
        added = molecule[self.n_atoms - 1:]
        group = Molecule.from_sites(added)
        offset = keep.sum()
        group_bonds = [
            (offset + group.index(b.site1), offset + group.index(b.site2))
            for b in group.get_covalent_bonds()
        ]
        new_bonds = [*group_bonds, (anchor - (anchor > index), offset)]
        self._bond_types = [
            *np.asarray(self.bond_types)[~removed], *(["S"] * len(new_bonds)),
        ]
        self._bonds = np.concatenate([
            bonds[~removed] - (bonds[~removed] > index), new_bonds,
        ])
        self.atoms = self.atoms[keep] + Atoms(
            [site.specie.symbol for site in added], positions=group.cart_coords,
        )
        self.port_idx = self.port_idx - (self.port_idx > index)
        self.atoms.info.update(
            port_idx=self.port_idx.tolist(), bonds=self._bonds,
            bond_types=self._bond_types,
        )
        self.__dict__.pop("port_descriptors", None)
        return self

    @property
    def centroid(self):
        """Centroid of the building block's X ports."""
        return np.mean(self.port_pos, axis=0)

    @property
    def port_pos(self):
        """Cartesian positions of the X ports, indexed by ``port_idx``."""
        return self.atoms[self.port_idx].positions

    def edge_frame(
        self,
        radial_tolerance=0.1,
        pca_gap_tolerance=0.1,
        frame_epsilon=1e-12
    ):
        '''
        Build a local coordinate frame for an edge with two X ports.

        The z axis points from the first X port to the second. The real atoms
        define the transverse axes through principal component analysis.

        Args:
            radial_tolerance (float):
                Transverse extent below which the edge is treated as linear, in
                angstroms.
            pca_gap_tolerance (float):
                Relative gap between transverse PCA eigenvalues below which torsion is
                ambiguous.

        Returns:
            An EdgeFrame with row-wise x, y, z axes and flags for ambiguous orientation.
        '''

        x0, x1 = self.port_pos
        edge_center = 0.5 * (x0 + x1)
        z = _unit_vector(x1 - x0, "the X0-to-X1 axis")

        real_pos = self.atoms.positions[self.atoms.numbers != 0]
        offsets = real_pos - edge_center
        axial = offsets @ z
        transverse = offsets - np.outer(axial, z)
        radial_extent = np.max(np.linalg.norm(transverse, axis=1))

        pca_pos = transverse - np.mean(transverse, axis=0)
        covariance = pca_pos.T @ pca_pos / len(pca_pos)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        if eigenvalues[-1] < frame_epsilon:
            mean_transverse = np.mean(transverse, axis=0)
            sign_degenerate = np.linalg.norm(mean_transverse) < frame_epsilon
            if sign_degenerate:
                x = self._arbitrary_transverse_axis(z)
            else:
                x = _unit_vector(mean_transverse, "the mean transverse axis")
            x -= np.dot(x, z) * z
            x = _unit_vector(x, "the mean transverse axis")
            y = _unit_vector(np.cross(z, x), "the edge-frame y axis")
            x = _unit_vector(np.cross(y, z), "the edge-frame x axis")
            return EdgeFrame(
                axes=np.stack([x, y, z]),
                pca_gap=0.0,
                torsion_degenerate=True,
                sign_degenerate=sign_degenerate,
            )

        x = eigenvectors[:, -1]
        x -= np.dot(x, z) * z
        x = _unit_vector(x, "the transverse PCA axis")
        centered_axial = axial - np.mean(axial)
        x, sign_degenerate = _orient_pca_axis(
            x,
            pca_pos,
            centered_axial,
        )

        pca_gap = float(
            (eigenvalues[-1] - eigenvalues[-2]) / eigenvalues[-1]
        )
        pca_degenerate = pca_gap < pca_gap_tolerance
        nearly_linear = radial_extent < radial_tolerance

        y = _unit_vector(np.cross(z, x), "the edge-frame y axis")
        x = _unit_vector(np.cross(y, z), "the edge-frame x axis")

        return EdgeFrame(
            axes=np.stack([x, y, z]),
            pca_gap=pca_gap,
            torsion_degenerate=nearly_linear or pca_degenerate,
            sign_degenerate=sign_degenerate,
        )

    @staticmethod
    def _arbitrary_transverse_axis(z):
        """Return a stable gauge axis for a geometrically linear edge."""

        cartesian_axes = np.eye(3)
        seed = cartesian_axes[np.argmin(np.abs(cartesian_axes @ z))]
        transverse = seed - np.dot(seed, z) * z
        return _unit_vector(transverse, "a transverse gauge axis")

    @property
    def conn_groups(self):
        """Real connection-atom indices per port: one atom or an oxygen pair."""
        groups = []
        numbers = self.atoms.numbers
        for x in self.port_idx:
            neighbors = tuple(
                b if a == x else a
                for a, b in self.bonds if a == x or b == x
            )
            valid = (len(neighbors) == 1 and numbers[neighbors[0]] != 0) or (
                len(neighbors) == 2 and all(numbers[i] == 8 for i in neighbors)
            )
            if not valid:
                raise ValueError("Each X port must have exactly one real connection atom "
                                 "or two oxygen connection atoms.")
            groups.append(neighbors)
        return tuple(groups)

    @property
    def conn_idx(self):
        """Connection-atom indices for single-atom ports; pairs use conn_groups."""
        groups = self.conn_groups
        if any(len(g) != 1 for g in groups):
            raise ValueError("Paired oxygen ports do not have a single connection-atom index.")
        return np.asarray([g[0] for g in groups], dtype=int)

    @property
    def conn_dists(self):
        """Maximum X-to-connection-atom distance per port, including paired sites."""
        return np.asarray([
            max(np.linalg.norm(point - self.atoms.positions[i]) for i in group)
            for point, group in zip(self.port_pos, self.conn_groups)
        ])

    @property
    def conn_symbols(self):
        """Connection-atom element per port; both atoms of a pair are oxygen."""
        symbols = self.atoms.get_chemical_symbols()
        return tuple(symbols[group[0]] for group in self.conn_groups)

    @cached_property
    def port_descriptors(self):
        """Return ``(connection element, max X-atom distance)`` for each X port."""
        return tuple(zip(self.conn_symbols, self.conn_dists))

    @property
    def n_ports(self):
        """Number of X ports (the fragment's coordination)."""
        return len(self.port_idx)

    @property
    def port_radii(self):
        """Distances from the centroid to each X port."""
        return np.linalg.norm(self.port_pos - self.centroid, axis=1)

    @property
    def is_edge(self):
        """Whether this fragment is an edge."""
        return self.n_ports == 2

    @property
    def kind(self):
        """Role and node chemistry, using the existing MOFid metal convention."""
        if self.is_edge:
            return "edge"
        if any(s != "X" and is_metal(s) for s in self.atoms.get_chemical_symbols()):
            return "metal_node"
        return "organic_node"

    @property
    def bonds(self):
        """Atom-index pairs describing the fragment's bonds."""
        if self._bonds is None:
            self.calc_bonds_dist()

        return self._bonds

    @property
    def bond_types(self):
        """Bond-order label for each entry in ``bonds``."""
        if self._bond_types is None:
            self.calc_bonds_dist()

        return self._bond_types

    @property
    def n_atoms(self):
        """Total number of atoms, including X ports."""
        return len(self.atoms)

    def calc_bonds_dist(self):
        '''
        Infer single bonds from distances and cache them on the block.

        Atoms are bonded when their distance is less than 1.2 times the sum
        of their ASE covalent radii. Coordinates are treated as a finite fragment.
        '''
        positions = self.atoms.positions
        cutoffs = 1.2 * np.asarray(natural_cutoffs(self.atoms))
        distances = np.linalg.norm(positions[:, None] - positions[None], axis=2)
        left, right = np.nonzero(np.triu(distances < cutoffs[:, None] + cutoffs, 1))
        self._bonds = np.column_stack((left, right))
        self._bond_types = ["S"] * len(self._bonds)

    def __repr__(self):
        return f"{type(self).__name__}: {self.name}, # of connection points: {self.n_ports}"


class Node(BuildingBlock):
    '''
    Load a node building block with at least one X port.

    Two-connected nodes keep their node role.

    Args:
        atoms (ase.Atoms):
            Atoms with building-block metadata; use read(path) to load a file.
    '''

    def __init__(self, atoms):
        '''
        Load a node building block with at least one X port.

        Args:
            atoms (ase.Atoms):
                Atoms with building-block metadata.
        '''
        super().__init__(atoms)
        if self.n_ports < 1:
            raise ValueError(f"Node {self.name!r} requires at least one X port.")

    @property
    def is_edge(self):
        '''
        Read the explicit topology role of this block.

        Returns:
            False, including for two-connected nodes.
        '''
        return False


class Edge(BuildingBlock):
    '''
    Load an edge with exactly two X ports.

    Args:
        atoms (ase.Atoms):
            Atoms with building-block metadata; use read(path) to load a file.
    '''

    def __init__(self, atoms):
        '''
        Load an edge with exactly two X ports.

        Args:
            atoms (ase.Atoms):
                Atoms with building-block metadata.
        '''
        super().__init__(atoms)
        if self.n_ports != 2:
            raise ValueError(f"Edge {self.name!r} requires exactly two X ports.")

    @property
    def is_edge(self):
        '''
        Read the explicit topology role of this block.

        Returns:
            True for an edge.
        '''
        return True
