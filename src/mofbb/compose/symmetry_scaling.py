"""Symmetry-reduced coordinates for topology scaling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pymatgen.symmetry.groups import SpaceGroup
from scipy.special import exprel


class Symmetry:
    """Small symmetry and linear-algebra helpers."""

    @staticmethod
    def diff(left, right):
        delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
        return delta - np.rint(delta)

    @staticmethod
    def key(operation, zero_tol=1.0e-12):
        rotation = np.rint(operation.rotation_matrix).astype(int)
        translation = np.mod(operation.translation_vector, 1.0)
        translation[np.isclose(translation, 1.0, atol=zero_tol)] = 0.0
        return (*rotation.ravel(), *np.round(translation, 12))

    @staticmethod
    @lru_cache(maxsize=None)
    def operations(symbol):
        return tuple(sorted(SpaceGroup(symbol).symmetry_ops, key=Symmetry.key))

    @staticmethod
    def null_space(constraints, columns, null_tol=1.0e-10):
        matrix = np.asarray(constraints, dtype=float).reshape(-1, columns)
        _, values, vectors = np.linalg.svd(matrix, full_matrices=True)
        scale = values[0] if len(values) else 0.0
        rank = np.count_nonzero(values > null_tol * max(1.0, scale))
        basis = vectors[rank:].T.copy()

        # SVD signs are arbitrary; normalize them for deterministic results.
        for column in basis.T:
            pivot = np.argmax(np.abs(column))
            if column[pivot] < 0.0:
                column *= -1.0
        return basis


@dataclass(frozen=True)
class SiteOrbit:
    """Cached affine expansion for one crystallographic site orbit."""

    tag: int
    representative: int
    member_indices: np.ndarray
    operation_indices: np.ndarray
    site_symmetry_operation_indices: np.ndarray
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
    """Parameterize a cell with point-group-invariant symmetric strains."""

    def __init__(self, initial_cell, rotations):
        self.initial_cell = np.asarray(initial_cell, dtype=float).copy()
        self.check_cell()
        self.rotations = self.unique(rotations)
        self.cartesian_rotations = self.cartesian()
        self.basis = self.strain_basis()

    def check_cell(self):
        if self.initial_cell.shape != (3, 3):
            raise ValueError("The topology cell must be a 3x3 matrix.")
        if not np.all(np.isfinite(self.initial_cell)):
            raise ValueError("The initial topology cell is non-finite.")
        if abs(np.linalg.det(self.initial_cell)) < np.finfo(float).eps:
            raise ValueError("The initial topology cell is singular.")

    @staticmethod
    def unique(rotations):
        unique = {}
        for rotation in rotations:
            rotation = np.asarray(rotation, dtype=float)
            key = tuple(np.rint(rotation).astype(int).ravel())
            unique.setdefault(key, rotation)
        return np.asarray(list(unique.values()))

    def cartesian(self, cell_tol=1.0e-10):
        metric = self.initial_cell @ self.initial_cell.T
        scale = max(1.0, float(np.max(np.abs(metric))))
        inverse = np.linalg.inv(self.initial_cell)
        result = []

        for index, rotation in enumerate(self.rotations):
            residual = np.max(np.abs(rotation.T @ metric @ rotation - metric))
            if residual > cell_tol * scale:
                raise ValueError(
                    "The initial cell is incompatible with declared point "
                    f"operation {index}: relative metric residual "
                    f"{residual / scale:.3e}."
                )

            cartesian = inverse @ rotation.T @ self.initial_cell
            residual = np.max(np.abs(cartesian @ cartesian.T - np.eye(3)))
            if residual > 5.0 * cell_tol:
                raise ValueError(
                    "The initial cell gives a non-orthogonal Cartesian "
                    f"representation for point operation {index}: residual "
                    f"{residual:.3e}."
                )
            result.append(cartesian)
        return np.asarray(result)

    def strain_basis(self):
        symmetric = np.zeros((6, 3, 3), dtype=float)
        symmetric[np.arange(3), np.arange(3), np.arange(3)] = 1.0
        value = 1.0 / np.sqrt(2.0)
        symmetric[3, 0, 1] = symmetric[3, 1, 0] = value
        symmetric[4, 0, 2] = symmetric[4, 2, 0] = value
        symmetric[5, 1, 2] = symmetric[5, 2, 1] = value

        constraints = []
        for rotation in self.cartesian_rotations:
            change = [rotation @ matrix @ rotation.T - matrix for matrix in symmetric]
            constraints.append(np.asarray(change).reshape(6, 9).T)

        coefficients = Symmetry.null_space(np.concatenate(constraints), 6)
        basis = np.einsum("ak,aij->kij", coefficients, symmetric)

        for index, rotation in enumerate(self.cartesian_rotations):
            transformed = np.asarray([rotation @ matrix @ rotation.T for matrix in basis])
            residual = np.max(np.abs(transformed - basis))
            if residual > 1.0e-9:
                raise ValueError(
                    "Failed to construct invariant strain basis for point "
                    f"operation {index}: residual {residual:.3e}."
                )
        return basis

    @property
    def n_parameters(self):
        return len(self.basis)

    def cell_and_derivatives(self, parameters):
        """Return the strained cell and its exact parameter derivatives."""

        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (self.n_parameters,):
            raise ValueError(
                f"Expected {self.n_parameters} cell parameters, got "
                f"shape {parameters.shape}."
            )

        strain = np.einsum("k,kij->ij", parameters, self.basis)
        if not np.all(np.isfinite(strain)):
            raise ValueError("The symmetry-reduced cell became non-finite.")

        values, vectors = np.linalg.eigh(strain)
        exponentials = np.exp(values)
        exponential = (vectors * exponentials) @ vectors.T
        cell = self.initial_cell @ exponential
        if not np.all(np.isfinite(cell)):
            raise ValueError("The symmetry-reduced cell became non-finite.")

        singular = np.linalg.svd(cell, compute_uv=False)
        if singular[-1] <= np.finfo(float).eps * max(1.0, singular[0]):
            raise ValueError("The symmetry-reduced cell became singular.")

        differences = values[:, None] - values[None, :]
        divided = exponentials[None, :] * exprel(differences)
        rotated = np.einsum("ia,kij,jb->kab", vectors, self.basis, vectors)
        frechet = np.einsum(
            "ia,kab,jb->kij", vectors, rotated * divided, vectors
        )
        derivatives = self.initial_cell @ frechet
        if not np.all(np.isfinite(derivatives)):
            raise ValueError(
                "The symmetry-reduced cell derivatives became non-finite."
            )
        return cell, derivatives

    @staticmethod
    def grad(cell_grad, derivatives):
        """Project a full cell gradient onto reduced strain parameters.

        For the cell matrix ``A(s)``, ``derivatives[k, i, j]`` stores
        ``d A[i, j] / d s[k]`` and ``cell_grad[i, j]`` stores
        ``d L / d A[i, j]``. The chain rule therefore gives the reduced
        gradient required by the optimizer:

        ``d L / d s[k] = sum_ij derivatives[k, i, j] * cell_grad[i, j]``.
        """

        return np.einsum("kij,ij->k", derivatives, cell_grad)


class SymmetryReducedParameterization:
    """Reduce topology nodes and cell to independent crystallographic DOFs."""

    def __init__(self, topology, mapping_tolerance=2.0e-5):
        if mapping_tolerance <= 0.0:
            raise ValueError("Symmetry mapping tolerance must be positive.")

        self.mapping_tolerance = float(mapping_tolerance)
        self.node_indices = np.asarray(topology.node_indices, dtype=int)
        if not len(self.node_indices):
            raise ValueError("A topology must contain at least one node.")

        try:
            self.operations = Symmetry.operations(topology.spacegroup)
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

        positions = topology.atoms.get_scaled_positions(wrap=True)
        self.initial_positions = np.asarray(positions[self.node_indices], dtype=float)
        self.tags = np.asarray(topology.node_types, dtype=int)[self.node_indices]
        self.operation_mappings = self.map_nodes()
        self.identity_operation_index = self.find_identity()
        self.orbits, self.position_bounds = self.build_orbits()
        self.n_node_parameters = sum(orbit.dimension for orbit in self.orbits)
        self.cell = InvariantCellParameterization(topology.atoms.cell, self.rotations)
        self.n_cell_parameters = self.cell.n_parameters
        self.n_parameters = self.n_node_parameters + self.n_cell_parameters

    def map_nodes(self):
        mappings = []
        node_count = len(self.node_indices)

        for operation_index, (rotation, translation) in enumerate(
            zip(self.rotations, self.translations)
        ):
            transformed = self.initial_positions @ rotation.T + translation
            differences = Symmetry.diff(
                transformed[:, None, :], self.initial_positions[None, :, :]
            )
            matches = np.all(np.abs(differences) <= self.mapping_tolerance, axis=2)
            mapping = np.empty(node_count, dtype=int)

            for local_index, candidates in enumerate(matches):
                targets = np.flatnonzero(candidates)
                slot = int(self.node_indices[local_index])
                if len(targets) == 0:
                    raise ValueError(
                        f"Declared symmetry operation {operation_index} cannot "
                        f"map node slot {slot} to an expanded node within "
                        f"fractional tolerance {self.mapping_tolerance:.1e}."
                    )
                if len(targets) > 1:
                    slots = self.node_indices[targets].tolist()
                    raise ValueError(
                        f"Declared symmetry operation {operation_index} maps node "
                        f"slot {slot} ambiguously to slots {slots} within "
                        f"fractional tolerance {self.mapping_tolerance:.1e}."
                    )

                target = int(targets[0])
                if self.tags[target] != self.tags[local_index]:
                    raise ValueError(
                        f"Declared symmetry operation {operation_index} maps node "
                        f"slot {slot} with tag {self.tags[local_index]} to slot "
                        f"{self.node_indices[target]} with tag {self.tags[target]}."
                    )
                mapping[local_index] = target

            if len(np.unique(mapping)) != node_count:
                raise ValueError(
                    f"Declared symmetry operation {operation_index} does not "
                    "map expanded node slots bijectively."
                )
            mappings.append(mapping)
        return np.asarray(mappings)

    def find_identity(self, zero_tol=1.0e-12):
        identity = np.eye(3, dtype=int)
        for index, (rotation, translation) in enumerate(
            zip(self.rotations, self.translations)
        ):
            is_rotation = np.array_equal(np.rint(rotation).astype(int), identity)
            is_translation = np.all(
                np.abs(Symmetry.diff(translation, 0.0)) <= zero_tol
            )
            if is_rotation and is_translation:
                return index
        raise ValueError(
            "The declared space group does not contain an identity operation."
        )

    def build_orbits(self):
        orbits, bounds = [], []
        start = 0
        for tag in np.unique(self.tags):
            orbit = self.make_orbit(int(tag), start)
            orbits.append(orbit)
            bounds.extend(orbit.bounds)
            start = orbit.parameter_slice.stop
        return tuple(orbits), tuple(bounds)

    def make_orbit(self, tag, start, zero_tol=1.0e-12):
        members = np.flatnonzero(self.tags == tag)
        members = members[np.argsort(self.node_indices[members])]
        representative = int(members[0])
        reached = np.unique(self.operation_mappings[:, representative])
        if not np.array_equal(np.sort(reached), np.sort(members)):
            raise ValueError(
                f"Node tag {tag} is not one crystallographic orbit: "
                f"representative slot {self.node_indices[representative]} reaches "
                f"{self.node_indices[reached].tolist()}, expected "
                f"{self.node_indices[members].tolist()}."
            )

        stabilizer = np.flatnonzero(
            self.operation_mappings[:, representative] == representative
        )
        constraints = np.concatenate(
            [self.rotations[index] - np.eye(3) for index in stabilizer]
        )
        basis = Symmetry.null_space(constraints, 3)
        basis[np.abs(basis) < zero_tol] = 0.0

        operation_indices, images, base_positions, jacobians = [], [], [], []
        representative_position = self.initial_positions[representative]
        for member in members:
            if member == representative:
                operation_index = self.identity_operation_index
            else:
                candidates = np.flatnonzero(
                    self.operation_mappings[:, representative] == member
                )
                if not len(candidates):
                    raise ValueError(
                        f"No declared operation maps representative node slot "
                        f"{self.node_indices[representative]} to same-tag slot "
                        f"{self.node_indices[member]}."
                    )
                operation_index = int(candidates[0])

            rotation = self.rotations[operation_index]
            unwrapped = rotation @ representative_position + self.translations[operation_index]
            image = np.rint(unwrapped - self.initial_positions[member]).astype(int)
            base = unwrapped - image
            residual = np.max(np.abs(base - self.initial_positions[member]))
            if residual > self.mapping_tolerance:
                raise ValueError(
                    f"Operation {operation_index} cannot reconstruct same-tag "
                    f"node slot {self.node_indices[member]} from representative "
                    f"slot {self.node_indices[representative]} with one fixed "
                    f"integer image: residual {residual:.3e}."
                )

            operation_indices.append(operation_index)
            images.append(image)
            base_positions.append(base)
            jacobians.append(rotation @ basis)

        jacobians = np.asarray(jacobians)
        bounds = self.image_bounds(jacobians, zero_tol=zero_tol)
        stop = start + basis.shape[1]
        return SiteOrbit(
            tag=tag,
            representative=int(self.node_indices[representative]),
            member_indices=members,
            operation_indices=np.asarray(operation_indices, dtype=int),
            site_symmetry_operation_indices=stabilizer,
            images=np.asarray(images, dtype=int),
            basis=basis,
            base_positions=np.asarray(base_positions),
            jacobians=jacobians,
            parameter_slice=slice(start, stop),
            bounds=bounds,
        )

    @staticmethod
    def image_bounds(
        jacobians,
        zero_tol=1.0e-12,
        image_margin=1.0e-8,
    ):
        """Bound amplitudes so each cached periodic image stays unique."""

        dimension = jacobians.shape[2]
        if dimension == 0:
            return ()

        active = np.abs(jacobians) >= zero_tol
        contributors = np.count_nonzero(active, axis=2)
        bounds = []

        for parameter in range(dimension):
            upper = np.inf
            for member, coordinate in np.argwhere(active[:, :, parameter]):
                coefficient = abs(jacobians[member, coordinate, parameter])
                share = contributors[member, coordinate]
                upper = min(
                    upper,
                    (0.5 - image_margin) / share / coefficient,
                )

            if not np.isfinite(upper) or upper <= 0.0:
                raise ValueError(
                    "A node displacement basis amplitude does not affect any "
                    "fractional coordinate."
                )
            bounds.append((-float(upper), float(upper)))

        for corner in range(1 << dimension):
            point = np.asarray(
                [bounds[index][(corner >> index) & 1] for index in range(dimension)]
            )
            displacement = np.einsum("mrd,d->mr", jacobians, point)
            if np.any(np.abs(displacement) >= 0.5):
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
        return parameters[: self.n_node_parameters], parameters[self.n_node_parameters :]

    def positions(self, node_parameters):
        """Expand orbit amplitudes into all node positions."""

        node_parameters = np.asarray(node_parameters, dtype=float)
        if node_parameters.shape != (self.n_node_parameters,):
            raise ValueError(
                f"Expected {self.n_node_parameters} node parameters, got "
                f"shape {node_parameters.shape}."
            )

        positions = np.empty_like(self.initial_positions)
        for orbit in self.orbits:
            values = node_parameters[orbit.parameter_slice]
            positions[orbit.member_indices] = orbit.base_positions + np.einsum(
                "mrd,d->mr", orbit.jacobians, values
            )
        return positions

    def pos_grad(self, y_grad):
        """Apply the transpose node-expansion Jacobian."""

        y_grad = np.asarray(y_grad, dtype=float)
        if y_grad.shape != self.initial_positions.shape:
            raise ValueError(
                f"Expected node gradient shape {self.initial_positions.shape}, "
                f"got {y_grad.shape}."
            )

        grad = np.empty(self.n_node_parameters, dtype=float)
        for orbit in self.orbits:
            grad[orbit.parameter_slice] = np.einsum(
                "mrd,mr->d",
                orbit.jacobians,
                y_grad[orbit.member_indices],
            )
        return grad
