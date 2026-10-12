"""Molecular building-block model."""
import copy
from dataclasses import dataclass
from functools import cached_property

import networkx as nx
import numpy as np

from ase.data import covalent_radii
from ase.neighborlist import natural_cutoffs

from mofbb.compose.chemistry import coordination_profile
from mofbb.compose.connectivity import bond_arrays
from mofbb.compose.local_structure import LocalStructure
from mofbb.compose.utils import read_building_block_xyz

_FRAME_EPSILON = 1e-12
_PCA_SIGN_TOLERANCE = 1e-8


@dataclass(frozen=True)
class EdgeFrame:
    """Right-handed intrinsic frame of a two-connected linker."""

    axes: np.ndarray
    pca_gap: float
    torsion_degenerate: bool
    sign_degenerate: bool

    @property
    def x(self):
        return self.axes[0]

    @property
    def y(self):
        return self.axes[1]

    @property
    def z(self):
        return self.axes[2]


def _unit_vector(vector, name):
    """Return ``vector`` normalized to unit length."""

    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if length < _FRAME_EPSILON:
        raise ValueError(f"Cannot define {name} from a near-zero vector.")
    return vector / length


def _orient_pca_axis(axis, transverse, axial):
    """Choose the sign of a PCA axis from the full atomic distribution."""

    projection = transverse @ axis
    scale = np.max(np.abs(projection))
    if scale < _FRAME_EPSILON:
        return axis, True

    p = projection / scale
    axial_scale = np.max(np.abs(axial))
    if axial_scale < _FRAME_EPSILON:
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
        score = float(score)
        if abs(score) >= _PCA_SIGN_TOLERANCE:
            return (axis if score > 0.0 else -axis), False

    for projection in p:
        if abs(projection) >= _PCA_SIGN_TOLERANCE:
            return (axis if projection > 0.0 else -axis), True
    return axis, True


class BuildingBlock:
    """Represent a molecular fragment placed onto a topology slot."""

    def __init__(self, bb_file, local_structure_func=None):
        """Read a building block from an ``.xyz`` file."""
        self.atoms = read_building_block_xyz(bb_file)
        self.name = self.atoms.info["name"]
        self.connection_point_indices = np.array(self.atoms.info["cpi"])
        self._bonds = self.atoms.info["bonds"]
        self._bond_types = self.atoms.info["bond_types"]
        self.local_structure_func = local_structure_func

    def copy(self):
        """Return an independent copy of this building block."""
        block = copy.copy(self)
        block.atoms = self.atoms.copy()
        block.connection_point_indices = self.connection_point_indices.copy()
        block._bonds = self.bonds.copy()
        block._bond_types = self.bond_types.copy()
        return block

    def validate_connectivity(self, bond_scale=1.2):
        """Require connected real atoms and pass the generation coordination screen."""
        bonds, _ = bond_arrays(self.n_atoms, self.bonds, np.zeros((len(self.bonds), 3)))
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
        """Screen known neighbors, leaving X partner identities for fusion.

        The graph analysis is cached in memory and ignores stored charges.
        Return the profile so assembly can reuse its neighbor counts.
        """
        profile = coordination_profile(self.atoms.numbers, self.bonds)
        if profile.error is not None:
            raise ValueError(f"{self.name!r}: {profile.error}")
        return profile

    def local_structure(self):
        """Build the ``LocalStructure`` of the connection points."""
        connection_points = self.atoms[self.connection_point_indices].positions
        return LocalStructure(
            connection_points,
            self.connection_point_indices,
            normalization_func=self.local_structure_func,
        )

    def set_centroid(self, centroid):
        """Rigidly translate the fragment to a new centroid."""
        positions = self.atoms.positions
        # Move centroid to zero.
        positions = positions - self.centroid
        # Recentroid by given value.
        positions = positions + centroid
        self.atoms.set_positions(positions)

    @property
    def centroid(self):
        """Geometric anchor of the building block."""
        centroid = np.mean(self.connection_points, axis=0)
        return centroid

    @property
    def connection_points(self):
        """Cartesian positions of the ``"X"`` connection points."""
        return self.atoms[self.connection_point_indices].positions

    def edge_frame(
        self,
        radial_tolerance=0.1,
        pca_gap_tolerance=0.1,
    ):
        """Construct a PCA-only intrinsic frame for an edge linker."""

        if not self.is_edge:
            raise ValueError(
                f"{self.name!r} has {self.n_connection_points} connection "
                "points; an edge frame requires exactly two."
            )
        if radial_tolerance < 0.0:
            raise ValueError("radial_tolerance must be non-negative.")
        if pca_gap_tolerance < 0.0:
            raise ValueError("pca_gap_tolerance must be non-negative.")

        x0, x1 = np.asarray(self.connection_points, dtype=float)
        edge_center = 0.5 * (x0 + x1)
        z = _unit_vector(x1 - x0, "the X0-to-X1 axis")

        symbols = np.asarray(self.atoms.get_chemical_symbols())
        real_pos = self.atoms.positions[symbols != "X"]
        if len(real_pos) == 0:
            raise ValueError(f"{self.name!r} contains no real atoms.")

        offsets = real_pos - edge_center
        axial = offsets @ z
        transverse = offsets - np.outer(axial, z)
        radial_extent = float(
            np.max(np.linalg.norm(transverse, axis=1))
        )

        pca_pos = transverse - np.mean(transverse, axis=0)
        covariance = pca_pos.T @ pca_pos / len(pca_pos)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        if eigenvalues[-1] < _FRAME_EPSILON:
            mean_transverse = np.mean(transverse, axis=0)
            sign_degenerate = np.linalg.norm(mean_transverse) < _FRAME_EPSILON
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

        leading_value = max(float(eigenvalues[-1]), _FRAME_EPSILON)
        pca_gap = float(
            (eigenvalues[-1] - eigenvalues[-2]) / leading_value
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
        """Return a stable gauge axis for a geometrically linear linker."""

        cartesian_axes = np.eye(3)
        seed = cartesian_axes[np.argmin(np.abs(cartesian_axes @ z))]
        transverse = seed - np.dot(seed, z) * z
        return _unit_vector(transverse, "a transverse gauge axis")

    @property
    def connection_point_anchor_groups(self):
        """Real anchors per port: one atom, or a bidentate oxygen pair."""
        groups = []
        numbers = self.atoms.numbers
        for x in self.connection_point_indices:
            neighbors = tuple(int(b if a == x else a)
                              for a, b in self.bonds if a == x or b == x)
            valid = (len(neighbors) == 1 and numbers[neighbors[0]] != 0) or (
                len(neighbors) == 2 and all(numbers[i] == 8 for i in neighbors)
            )
            if not valid:
                raise ValueError("Each X port must have exactly one real-atom anchor "
                                 "or two oxygen anchors.")
            groups.append(neighbors)
        return tuple(groups)

    @property
    def connection_point_anchor_indices(self):
        """Single anchors only; paired ports must use anchor_groups explicitly."""
        groups = self.connection_point_anchor_groups
        if any(len(g) != 1 for g in groups):
            raise ValueError("Paired oxygen ports do not have a single anchor index.")
        return np.asarray([g[0] for g in groups], dtype=int)

    @property
    def connection_point_anchor_lengths(self):
        """Maximum X-to-anchor distance, a conservative bound for paired sites."""
        return np.asarray([
            max(np.linalg.norm(point - self.atoms.positions[i]) for i in group)
            for point, group in zip(self.connection_points,
                                    self.connection_point_anchor_groups, strict=True)
        ])

    @property
    def connection_point_anchor_symbols(self):
        """Anchor element; both anchors of a paired port are oxygen."""
        symbols = self.atoms.get_chemical_symbols()
        return tuple(symbols[group[0]] for group in self.connection_point_anchor_groups)

    @cached_property
    def port_descriptors(self):
        """Return ``(anchor element, X-anchor length)`` for each X site."""
        return tuple(
            zip(
                self.connection_point_anchor_symbols,
                self.connection_point_anchor_lengths,
                strict=True,
            )
        )

    @property
    def n_connection_points(self):
        """Number of connection points (the fragment's coordination)."""
        return len(self.connection_point_indices)

    @property
    def lengths(self):
        """Distances from the centroid to each connection point."""
        dists = self.connection_points - self.centroid
        norms = np.linalg.norm(dists, axis=1)
        return norms

    @property
    def is_edge(self):
        """Whether this fragment is an edge linker."""
        return self.n_connection_points == 2

    @property
    def bonds(self):
        """Atom-index pairs describing the fragment's bonds."""
        if self._bonds is None:
            self.calculate_bonds()

        return self._bonds

    @property
    def bond_types(self):
        """Bond-order label for each entry in ``bonds``."""
        if self._bond_types is None:
            self.calculate_bonds()

        return self._bond_types

    @property
    def n_atoms(self):
        """Total number of atoms, including connection points."""
        return len(self.atoms)

    def calculate_bonds(self):
        """Infer bonds from interatomic distances."""
        positions = self.atoms.positions
        cutoffs = 1.2 * np.asarray(natural_cutoffs(self.atoms))
        distances = np.linalg.norm(positions[:, None] - positions[None], axis=2)
        left, right = np.nonzero(np.triu(distances < cutoffs[:, None] + cutoffs, 1))
        self._bonds = np.column_stack((left, right))
        self._bond_types = ["S"] * len(self._bonds)

    def __repr__(self):
        return f"BuildingBlock: {self.name}, # of connection points: {self.n_connection_points}"
