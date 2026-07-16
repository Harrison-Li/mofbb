"""Symmetry-reduced coordinates for topology scaling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.special
from pymatgen.symmetry.groups import SpaceGroup


_MAPPING_TOLERANCE = 2.0e-5
_CELL_TOLERANCE = 1.0e-10
_NULL_SPACE_TOLERANCE = 1.0e-10
_BASIS_ZERO_TOLERANCE = 1.0e-12
_IMAGE_MARGIN = 1.0e-8


def _periodic_difference(left, right):
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return difference - np.rint(difference)


def _operation_key(operation):
    rotation = np.rint(operation.rotation_matrix).astype(int)
    translation = np.mod(operation.translation_vector, 1.0)
    translation[np.isclose(translation, 1.0, atol=1.0e-12)] = 0.0
    return (*rotation.reshape(-1), *np.round(translation, decimals=12))


@lru_cache(maxsize=None)
def _spacegroup_operations(symbol):
    return tuple(sorted(SpaceGroup(symbol).symmetry_ops, key=_operation_key))


def _null_space(constraints, n_columns):
    constraints = np.asarray(constraints, dtype=float).reshape(-1, n_columns)
    _, singular_values, right_vectors = np.linalg.svd(
        constraints,
        full_matrices=True,
    )
    scale = singular_values[0] if len(singular_values) else 0.0
    rank = int(
        np.count_nonzero(
            singular_values > _NULL_SPACE_TOLERANCE * max(1.0, scale)
        )
    )
    basis = right_vectors[rank:].T.copy()

    # SVD signs are arbitrary. Fix them so operation maps and tests are
    # deterministic across LAPACK implementations.
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return basis


@dataclass(frozen=True)
class NodeOrbit:
    """Cached affine expansion for one crystallographic node orbit."""

    tag: int
    representative: int
    member_indices: np.ndarray
    operation_indices: np.ndarray
    stabilizer_operation_indices: np.ndarray
    images: np.ndarray
    basis: np.ndarray
    base_positions: np.ndarray
    jacobians: np.ndarray
    parameter_slice: slice
    bounds: tuple[tuple[float, float], ...]

    @property
    def dimension(self):
        return self.basis.shape[1]


class InvariantCellParameterization:
    """Parameterize a cell by point-group-invariant symmetric strains."""

    def __init__(self, initial_cell, rotations):
        self.initial_cell = np.asarray(initial_cell, dtype=float).copy()
        if self.initial_cell.shape != (3, 3):
            raise ValueError("The topology cell must be a 3x3 matrix.")
        if not np.all(np.isfinite(self.initial_cell)):
            raise ValueError("The initial topology cell is non-finite.")
        if abs(np.linalg.det(self.initial_cell)) < np.finfo(float).eps:
            raise ValueError("The initial topology cell is singular.")

        unique_rotations = []
        seen = set()
        for rotation in rotations:
            rotation = np.asarray(rotation, dtype=float)
            key = tuple(np.rint(rotation).astype(int).reshape(-1))
            if key not in seen:
                seen.add(key)
                unique_rotations.append(rotation)
        self.rotations = np.asarray(unique_rotations)

        metric = self.initial_cell @ self.initial_cell.T
        metric_scale = max(1.0, float(np.max(np.abs(metric))))
        inverse_cell = np.linalg.inv(self.initial_cell)
        cartesian_rotations = []
        for index, rotation in enumerate(self.rotations):
            transformed_metric = rotation.T @ metric @ rotation
            residual = float(np.max(np.abs(transformed_metric - metric)))
            if residual > _CELL_TOLERANCE * metric_scale:
                raise ValueError(
                    "The initial cell is incompatible with declared point "
                    f"operation {index}: relative metric residual "
                    f"{residual / metric_scale:.3e}."
                )

            cartesian_rotation = (
                inverse_cell @ rotation.T @ self.initial_cell
            )
            orthogonality_residual = float(
                np.max(
                    np.abs(
                        cartesian_rotation @ cartesian_rotation.T - np.eye(3)
                    )
                )
            )
            if orthogonality_residual > 5.0 * _CELL_TOLERANCE:
                raise ValueError(
                    "The initial cell gives a non-orthogonal Cartesian "
                    f"representation for point operation {index}: residual "
                    f"{orthogonality_residual:.3e}."
                )
            cartesian_rotations.append(cartesian_rotation)
        self.cartesian_rotations = np.asarray(cartesian_rotations)

        symmetric_basis = np.zeros((6, 3, 3), dtype=float)
        symmetric_basis[0, 0, 0] = 1.0
        symmetric_basis[1, 1, 1] = 1.0
        symmetric_basis[2, 2, 2] = 1.0
        symmetric_basis[3, 0, 1] = symmetric_basis[3, 1, 0] = 1.0 / np.sqrt(2.0)
        symmetric_basis[4, 0, 2] = symmetric_basis[4, 2, 0] = 1.0 / np.sqrt(2.0)
        symmetric_basis[5, 1, 2] = symmetric_basis[5, 2, 1] = 1.0 / np.sqrt(2.0)

        constraints = []
        for rotation in self.cartesian_rotations:
            transformed = np.array(
                [rotation @ matrix @ rotation.T - matrix for matrix in symmetric_basis]
            )
            constraints.append(transformed.reshape(6, 9).T)
        coefficient_basis = _null_space(np.concatenate(constraints), 6)
        self.basis = np.einsum(
            "ak,aij->kij",
            coefficient_basis,
            symmetric_basis,
        )

        for index, rotation in enumerate(self.cartesian_rotations):
            residual = np.max(
                np.abs(
                    np.array(
                        [rotation @ matrix @ rotation.T for matrix in self.basis]
                    )
                    - self.basis
                )
            )
            if residual > 1.0e-9:
                raise ValueError(
                    "Failed to construct invariant strain basis for point "
                    f"operation {index}: residual {residual:.3e}."
                )

    @property
    def n_parameters(self):
        return len(self.basis)

    def cell_and_derivatives(self, parameters):
        """Return the strained cell and exact derivatives with respect to it."""

        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (self.n_parameters,):
            raise ValueError(
                f"Expected {self.n_parameters} cell parameters, got "
                f"shape {parameters.shape}."
            )
        strain = np.einsum("k,kij->ij", parameters, self.basis)
        if not np.all(np.isfinite(strain)):
            raise ValueError("The symmetry-reduced cell became non-finite.")

        # E is symmetric, so one eigendecomposition gives both exp(E) and
        # every exact Frechet derivative. In the eigenbasis, the derivative
        # multiplies each direction by the exponential divided-difference
        # matrix. scipy.special.exprel evaluates that matrix stably when two
        # eigenvalues coincide.
        eigenvalues, eigenvectors = np.linalg.eigh(strain)
        exponential_values = np.exp(eigenvalues)
        exponential = (
            eigenvectors * exponential_values
        ) @ eigenvectors.T
        cell = self.initial_cell @ exponential
        if not np.all(np.isfinite(cell)):
            raise ValueError("The symmetry-reduced cell became non-finite.")
        singular_values = np.linalg.svd(cell, compute_uv=False)
        if singular_values[-1] <= np.finfo(float).eps * max(
            1.0, singular_values[0]
        ):
            raise ValueError("The symmetry-reduced cell became singular.")

        eigenvalue_differences = (
            eigenvalues[:, np.newaxis] - eigenvalues[np.newaxis, :]
        )
        divided_differences = exponential_values[np.newaxis, :] * (
            scipy.special.exprel(eigenvalue_differences)
        )
        rotated_basis = np.einsum(
            "ia,kij,jb->kab",
            eigenvectors,
            self.basis,
            eigenvectors,
        )
        frechet_derivatives = np.einsum(
            "ia,kab,jb->kij",
            eigenvectors,
            rotated_basis * divided_differences,
            eigenvectors,
        )
        derivatives = self.initial_cell @ frechet_derivatives
        if not np.all(np.isfinite(derivatives)):
            raise ValueError(
                "The symmetry-reduced cell derivatives became non-finite."
            )
        return cell, derivatives

    @staticmethod
    def gradient(cell_gradient, derivatives):
        """Project a full Cartesian cell gradient onto strain variables."""

        return np.einsum("kij,ij->k", derivatives, cell_gradient)


class SymmetryReducedParameterization:
    """Reduce topology nodes and cell to independent crystallographic DOFs."""

    def __init__(self, topology, mapping_tolerance=_MAPPING_TOLERANCE):
        if mapping_tolerance <= 0.0:
            raise ValueError("Symmetry mapping tolerance must be positive.")
        self.mapping_tolerance = float(mapping_tolerance)
        self.node_indices = np.asarray(topology.node_indices, dtype=int)
        if len(self.node_indices) == 0:
            raise ValueError("A topology must contain at least one node.")

        try:
            self.operations = _spacegroup_operations(topology.spacegroup)
        except Exception as error:
            raise ValueError(
                f"Cannot construct declared space group {topology.spacegroup!r}."
            ) from error
        self.rotations = np.asarray(
            [operation.rotation_matrix for operation in self.operations],
            dtype=float,
        )
        self.translations = np.asarray(
            [operation.translation_vector for operation in self.operations],
            dtype=float,
        )

        all_positions = topology.atoms.get_scaled_positions(wrap=True)
        self.initial_positions = np.asarray(
            all_positions[self.node_indices],
            dtype=float,
        )
        all_tags = np.asarray(topology.node_types, dtype=int)
        self.tags = all_tags[self.node_indices]
        self.operation_mappings = self._validate_operation_mappings()

        self.identity_operation_index = self._find_identity_operation()
        self.orbits, self.position_bounds = self._build_orbits()
        self.n_node_parameters = sum(orbit.dimension for orbit in self.orbits)
        self.cell = InvariantCellParameterization(
            np.asarray(topology.atoms.cell, dtype=float),
            self.rotations,
        )
        self.n_cell_parameters = self.cell.n_parameters
        self.n_parameters = self.n_node_parameters + self.n_cell_parameters

    def _validate_operation_mappings(self):
        mappings = []
        slot_count = len(self.node_indices)
        for operation_index, (rotation, translation) in enumerate(
            zip(self.rotations, self.translations)
        ):
            transformed = self.initial_positions @ rotation.T + translation
            differences = _periodic_difference(
                transformed[:, np.newaxis, :],
                self.initial_positions[np.newaxis, :, :],
            )
            matches = np.all(
                np.abs(differences) <= self.mapping_tolerance,
                axis=2,
            )
            mapping = np.empty(slot_count, dtype=int)
            for local_index, candidates in enumerate(matches):
                candidate_indices = np.flatnonzero(candidates)
                slot_index = int(self.node_indices[local_index])
                if len(candidate_indices) == 0:
                    raise ValueError(
                        f"Declared symmetry operation {operation_index} cannot "
                        f"map node slot {slot_index} to an expanded node within "
                        f"fractional tolerance {self.mapping_tolerance:.1e}."
                    )
                if len(candidate_indices) > 1:
                    candidate_slots = self.node_indices[candidate_indices].tolist()
                    raise ValueError(
                        f"Declared symmetry operation {operation_index} maps node "
                        f"slot {slot_index} ambiguously to slots {candidate_slots} "
                        f"within fractional tolerance {self.mapping_tolerance:.1e}."
                    )
                target = int(candidate_indices[0])
                if self.tags[target] != self.tags[local_index]:
                    raise ValueError(
                        f"Declared symmetry operation {operation_index} maps node "
                        f"slot {slot_index} with tag {self.tags[local_index]} to "
                        f"slot {self.node_indices[target]} with tag "
                        f"{self.tags[target]}."
                    )
                mapping[local_index] = target
            if len(np.unique(mapping)) != slot_count:
                raise ValueError(
                    f"Declared symmetry operation {operation_index} does not "
                    "map expanded node slots bijectively."
                )
            mappings.append(mapping)
        return np.asarray(mappings)

    def _find_identity_operation(self):
        for index, (rotation, translation) in enumerate(
            zip(self.rotations, self.translations)
        ):
            identity_rotation = np.array_equal(
                np.rint(rotation).astype(int),
                np.eye(3, dtype=int),
            )
            identity_translation = np.all(
                np.abs(_periodic_difference(translation, 0.0)) <= 1.0e-12
            )
            if identity_rotation and identity_translation:
                return index
        raise ValueError(
            "The declared space group does not contain an identity operation."
        )

    def _build_orbits(self):
        orbits = []
        bounds = []
        parameter_start = 0
        for tag in np.unique(self.tags):
            members = np.flatnonzero(self.tags == tag)
            members = members[np.argsort(self.node_indices[members])]
            representative = int(members[0])
            reached = np.unique(self.operation_mappings[:, representative])
            if not np.array_equal(np.sort(reached), np.sort(members)):
                reached_slots = self.node_indices[reached].tolist()
                expected_slots = self.node_indices[members].tolist()
                raise ValueError(
                    f"Node tag {int(tag)} is not one crystallographic orbit: "
                    f"representative slot {self.node_indices[representative]} "
                    f"reaches {reached_slots}, expected {expected_slots}."
                )

            stabilizer = np.flatnonzero(
                self.operation_mappings[:, representative] == representative
            )
            constraints = np.concatenate(
                [self.rotations[index] - np.eye(3) for index in stabilizer]
            )
            basis = _null_space(constraints, 3)
            basis[np.abs(basis) < _BASIS_ZERO_TOLERANCE] = 0.0
            dimension = basis.shape[1]

            operation_indices = []
            images = []
            base_positions = []
            jacobians = []
            representative_position = self.initial_positions[representative]
            for member in members:
                if member == representative:
                    operation_index = self.identity_operation_index
                else:
                    candidates = np.flatnonzero(
                        self.operation_mappings[:, representative] == member
                    )
                    if len(candidates) == 0:
                        raise ValueError(
                            f"No declared operation maps representative node slot "
                            f"{self.node_indices[representative]} to same-tag slot "
                            f"{self.node_indices[member]}."
                        )
                    operation_index = int(candidates[0])

                rotation = self.rotations[operation_index]
                translation = self.translations[operation_index]
                unwrapped = rotation @ representative_position + translation
                image = np.rint(
                    unwrapped - self.initial_positions[member]
                ).astype(int)
                base_position = unwrapped - image
                residual = float(
                    np.max(
                        np.abs(base_position - self.initial_positions[member])
                    )
                )
                if residual > self.mapping_tolerance:
                    raise ValueError(
                        f"Operation {operation_index} cannot reconstruct same-tag "
                        f"node slot {self.node_indices[member]} from representative "
                        f"slot {self.node_indices[representative]} with one fixed "
                        f"integer image: residual {residual:.3e}."
                    )
                operation_indices.append(operation_index)
                images.append(image)
                base_positions.append(base_position)
                jacobians.append(rotation @ basis)

            jacobians = np.asarray(jacobians)
            orbit_bounds = self._fixed_image_bounds(jacobians)
            parameter_stop = parameter_start + dimension
            parameter_slice = slice(parameter_start, parameter_stop)
            bounds.extend(orbit_bounds)
            orbits.append(
                NodeOrbit(
                    tag=int(tag),
                    representative=int(self.node_indices[representative]),
                    member_indices=members,
                    operation_indices=np.asarray(operation_indices, dtype=int),
                    stabilizer_operation_indices=stabilizer,
                    images=np.asarray(images, dtype=int),
                    basis=basis,
                    base_positions=np.asarray(base_positions),
                    jacobians=jacobians,
                    parameter_slice=parameter_slice,
                    bounds=orbit_bounds,
                )
            )
            parameter_start = parameter_stop
        return tuple(orbits), tuple(bounds)

    @staticmethod
    def _fixed_image_bounds(jacobians):
        """Build a box that keeps every cached periodic image unique.

        Orbit positions are unwrapped affine functions of the amplitudes. For
        every member coordinate, divide the available half-period displacement
        among the amplitudes that affect it. This yields independent bounds
        for L-BFGS-B while guaranteeing that every box corner moves every
        generated component by strictly less than half a lattice period.
        Consequently, the integer image selected from the initial topology
        remains the unique nearest image throughout optimization.
        """

        dimension = jacobians.shape[2]
        if dimension == 0:
            return ()

        active = np.abs(jacobians) >= _BASIS_ZERO_TOLERANCE
        contributors = np.count_nonzero(active, axis=2)
        bounds = []
        for parameter_index in range(dimension):
            member_coordinates = np.argwhere(active[:, :, parameter_index])
            upper = np.inf
            for member, coordinate in member_coordinates:
                coefficient = abs(jacobians[member, coordinate, parameter_index])
                share_count = contributors[member, coordinate]
                allocated_displacement = (
                    0.5 - _IMAGE_MARGIN
                ) / share_count
                upper = min(upper, allocated_displacement / coefficient)

            if not np.isfinite(upper) or upper <= 0.0:
                raise ValueError(
                    "A node displacement basis amplitude does not affect any "
                    "fractional coordinate."
                )
            bounds.append((-float(upper), float(upper)))

        # There are at most three amplitudes per orbit, so verify the entire
        # derived box for every orbit member once.
        for corner_index in range(1 << dimension):
            corner = np.array(
                [
                    bounds[index][(corner_index >> index) & 1]
                    for index in range(dimension)
                ]
            )
            displacements = np.einsum("mrd,d->mr", jacobians, corner)
            if np.any(np.abs(displacements) >= 0.5):
                raise ValueError(
                    "Derived node bounds do not preserve fixed periodic images."
                )
        return tuple(bounds)

    def split(self, parameters):
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (self.n_parameters,):
            raise ValueError(
                f"Expected {self.n_parameters} reduced parameters, got "
                f"shape {parameters.shape}."
            )
        return (
            parameters[: self.n_node_parameters],
            parameters[self.n_node_parameters :],
        )

    def positions(self, node_parameters):
        """Expand independent orbit amplitudes into all node positions."""

        node_parameters = np.asarray(node_parameters, dtype=float)
        if node_parameters.shape != (self.n_node_parameters,):
            raise ValueError(
                f"Expected {self.n_node_parameters} node parameters, got "
                f"shape {node_parameters.shape}."
            )
        positions = np.empty_like(self.initial_positions)
        for orbit in self.orbits:
            amplitudes = node_parameters[orbit.parameter_slice]
            positions[orbit.member_indices] = orbit.base_positions + np.einsum(
                "mrd,d->mr",
                orbit.jacobians,
                amplitudes,
            )
        return positions

    def position_gradient(self, full_gradient):
        """Apply the transpose node-expansion Jacobian."""

        full_gradient = np.asarray(full_gradient, dtype=float)
        if full_gradient.shape != self.initial_positions.shape:
            raise ValueError(
                f"Expected node gradient shape {self.initial_positions.shape}, "
                f"got {full_gradient.shape}."
            )
        gradient = np.empty(self.n_node_parameters, dtype=float)
        for orbit in self.orbits:
            gradient[orbit.parameter_slice] = np.einsum(
                "mrd,mr->d",
                orbit.jacobians,
                full_gradient[orbit.member_indices],
            )
        return gradient
