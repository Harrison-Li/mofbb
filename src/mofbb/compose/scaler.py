"""Node, lattice, and rigid-pose optimization for framework composition.

Includes symmetry-constrained parameterizations, node/lattice and joint geometry
optimizers, and the optional fixed-cell framework fitting stage.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.groups import SpaceGroup
from scipy.optimize import OptimizeResult, least_squares, minimize
from scipy.spatial.transform import Rotation
from scipy.special import exprel

from mofbb.compose.placement.errors import SymmetryCompatibilityError
from mofbb.compose.placement.ports import DirectEdgeRepair, PortAssignmentSolver
from mofbb.compose.utils import wrap_fractional


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
    def _ordered_operation_data(raw):
        ordered = tuple(sorted(raw, key=Symmetry.key))
        ordered_indices = {
            Symmetry.key(operation): index
            for index, operation in enumerate(ordered)
        }
        raw_to_ordered = tuple(
            ordered_indices[Symmetry.key(operation)] for operation in raw
        )
        return raw, ordered, raw_to_ordered

    @staticmethod
    @lru_cache(maxsize=None)
    def operation_data(symbol):
        return Symmetry._ordered_operation_data(
            tuple(SpaceGroup(symbol).symmetry_ops)
        )

    @staticmethod
    def topology_operation_data(topology):
        declared = topology.atoms.info.get("symmetry_operations")
        if not declared:
            return Symmetry.operation_data(topology.spacegroup)
        raw = tuple(
            SymmOp.from_rotation_and_translation(
                operation["rotation"], operation["translation"]
            )
            for operation in declared
        )
        return Symmetry._ordered_operation_data(raw)

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


class SymmetryInvariantLatticeParameterization:
    """Parameterize a lattice with point-group-invariant symmetric strains."""

    def __init__(self, initial_lattice, rotations):
        self.initial_lattice = np.asarray(initial_lattice, dtype=float).copy()
        if not np.all(np.isfinite(self.initial_lattice)):
            raise ValueError("The initial topology lattice is non-finite.")
        if abs(np.linalg.det(self.initial_lattice)) < np.finfo(float).eps:
            raise ValueError("The initial topology lattice is singular.")
        self.rotations = self.unique(rotations)
        self.cartesian_rotations = self.cartesian()
        self.basis = self.strain_basis()

    @staticmethod
    def unique(rotations):
        unique = {}
        for rotation in rotations:
            rotation = np.asarray(rotation, dtype=float)
            key = tuple(np.rint(rotation).astype(int).ravel())
            unique.setdefault(key, rotation)
        return np.asarray(list(unique.values()))

    def cartesian(self, lattice_tol=1.0e-10):
        metric = self.initial_lattice @ self.initial_lattice.T
        scale = max(1.0, float(np.max(np.abs(metric))))
        inverse = np.linalg.inv(self.initial_lattice)
        result = []

        for index, rotation in enumerate(self.rotations):
            residual = np.max(np.abs(rotation.T @ metric @ rotation - metric))
            if residual > lattice_tol * scale:
                raise ValueError(
                    "The initial lattice is incompatible with declared point "
                    f"operation {index}: relative metric residual "
                    f"{residual / scale:.3e}."
                )

            cartesian = inverse @ rotation.T @ self.initial_lattice
            residual = np.max(np.abs(cartesian @ cartesian.T - np.eye(3)))
            if residual > 5.0 * lattice_tol:
                raise ValueError(
                    "The initial lattice gives a non-orthogonal Cartesian "
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

    def lattice_and_derivatives(self, parameters):
        """Return the strained lattice and its exact parameter derivatives."""

        parameters = np.asarray(parameters, dtype=float)
        strain = np.einsum("k,kij->ij", parameters, self.basis)
        if not np.all(np.isfinite(strain)):
            raise ValueError("The symmetry-constrained lattice became non-finite.")

        values, vectors = np.linalg.eigh(strain)
        exponentials = np.exp(values)
        exponential = (vectors * exponentials) @ vectors.T
        lattice = self.initial_lattice @ exponential
        if not np.all(np.isfinite(lattice)):
            raise ValueError("The symmetry-constrained lattice became non-finite.")

        singular = np.linalg.svd(lattice, compute_uv=False)
        if singular[-1] <= np.finfo(float).eps * max(1.0, singular[0]):
            raise ValueError("The symmetry-constrained lattice became singular.")

        differences = values[:, None] - values[None, :]
        divided = exponentials[None, :] * exprel(differences)
        rotated = np.einsum("ia,kij,jb->kab", vectors, self.basis, vectors)
        frechet = np.einsum(
            "ia,kab,jb->kij", vectors, rotated * divided, vectors
        )
        derivatives = self.initial_lattice @ frechet
        if not np.all(np.isfinite(derivatives)):
            raise ValueError(
                "The symmetry-constrained lattice derivatives became non-finite."
            )
        return lattice, derivatives

    @staticmethod
    def gradient(lattice_gradient, derivatives):
        """Project a full lattice gradient onto reduced strain parameters."""

        return np.einsum("kij,ij->k", derivatives, lattice_gradient)


class SymmetryConstrainedNodeLatticeParameterization:
    """Parameterize independent topological-node and lattice degrees of freedom."""

    def __init__(self, topology, mapping_tolerance=2.0e-5):
        self.mapping_tolerance = float(mapping_tolerance)
        self.node_indices = np.asarray(topology.node_indices, dtype=int)
        self.operations = Symmetry.topology_operation_data(topology)[1]

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
        self.lattice = SymmetryInvariantLatticeParameterization(
            topology.atoms.cell,
            self.rotations,
        )
        self.n_lattice_parameters = self.lattice.n_parameters
        self.n_parameters = self.n_node_parameters + self.n_lattice_parameters

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
            invalid = np.flatnonzero(matches.sum(axis=1) != 1)
            if len(invalid):
                slot = int(invalid[0])
                targets = self.node_indices[np.flatnonzero(matches[slot])].tolist()
                raise ValueError(
                    f"Declared symmetry operation {operation_index} must map node slot "
                    f"{self.node_indices[slot]} uniquely within fractional tolerance "
                    f"{self.mapping_tolerance:.1e}; matched slots {targets}."
                )
            mapping = np.argmax(matches, axis=1)
            if np.any(self.tags != self.tags[mapping]):
                raise ValueError(f"Declared symmetry operation {operation_index} changes node tags.")
            if len(np.unique(mapping)) != node_count:
                raise ValueError(
                    f"Declared symmetry operation {operation_index} is not bijective on node slots."
                )
            mappings.append(mapping)
        return np.asarray(mappings)

    def find_identity(self, zero_tol=1.0e-12):
        identity = np.all(np.rint(self.rotations) == np.eye(3), axis=(1, 2))
        identity &= np.all(abs(Symmetry.diff(self.translations, 0.0)) <= zero_tol, axis=1)
        indices = np.flatnonzero(identity)
        if not len(indices):
            raise ValueError("The declared space group does not contain an identity operation.")
        return int(indices[0])

    def build_orbits(self):
        orbits, bounds = [], []
        start = 0
        for tag in np.unique(self.tags):
            remaining = set(np.flatnonzero(self.tags == tag))
            while remaining:
                representative = min(remaining)
                members = np.unique(
                    self.operation_mappings[:, representative]
                )
                if not set(members) <= remaining:
                    raise ValueError(
                        f"Overlapping crystallographic orbits for node tag {tag}."
                    )
                orbit = self.make_orbit(int(tag), start, members)
                orbits.append(orbit)
                bounds.extend(orbit.bounds)
                start = orbit.parameter_slice.stop
                remaining.difference_update(members)
        return tuple(orbits), tuple(bounds)

    def make_orbit(self, tag, start, members, zero_tol=1.0e-12):
        members = np.asarray(members, dtype=int)
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
        basis = Symmetry.null_space(self.rotations[stabilizer] - np.eye(3), 3)
        basis[np.abs(basis) < zero_tol] = 0.0
        mapping = self.operation_mappings[:, representative]
        operation_indices = np.argmax(mapping[:, None] == members, axis=0)
        operation_indices[members == representative] = self.identity_operation_index
        rotations = self.rotations[operation_indices]
        unwrapped = rotations @ self.initial_positions[representative] + self.translations[operation_indices]
        images = np.rint(unwrapped - self.initial_positions[members]).astype(int)
        jacobians = rotations @ basis
        bounds = self.image_bounds(jacobians, zero_tol=zero_tol)
        stop = start + basis.shape[1]
        return SiteOrbit(
            tag=tag,
            representative=int(self.node_indices[representative]),
            member_indices=members,
            operation_indices=operation_indices,
            site_symmetry_operation_indices=stabilizer,
            images=images,
            basis=basis,
            base_positions=unwrapped - images,
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

        coefficients = np.abs(jacobians)
        active = coefficients >= zero_tol
        shares = np.maximum(np.count_nonzero(active, axis=2), 1)
        limits = np.full_like(coefficients, np.inf)
        np.divide((0.5 - image_margin) / shares[..., None], coefficients,
                  out=limits, where=active)
        upper = np.min(limits, axis=(0, 1))
        if not np.isfinite(upper).all():
            raise ValueError("A node displacement basis amplitude does not affect any fractional coordinate.")
        return tuple((-float(value), float(value)) for value in upper)

    def split(self, parameters):
        parameters = np.asarray(parameters, dtype=float)
        return parameters[: self.n_node_parameters], parameters[self.n_node_parameters :]

    def positions(self, node_parameters):
        """Expand orbit amplitudes into all node positions."""

        node_parameters = np.asarray(node_parameters, dtype=float)
        positions = np.empty_like(self.initial_positions)
        for orbit in self.orbits:
            values = node_parameters[orbit.parameter_slice]
            positions[orbit.member_indices] = orbit.base_positions + np.einsum(
                "mrd,d->mr", orbit.jacobians, values
            )
        return positions

    def position_gradient(self, position_gradient):
        """Apply the transpose node-expansion Jacobian."""

        position_gradient = np.asarray(position_gradient, dtype=float)
        grad = np.empty(self.n_node_parameters, dtype=float)
        for orbit in self.orbits:
            grad[orbit.parameter_slice] = np.einsum(
                "mrd,mr->d",
                orbit.jacobians,
                position_gradient[orbit.member_indices],
            )
        return grad


def _objective_value_and_gradient(
    fractional_positions,
    lattice,
    ij,
    ik,
    ij_image,
    ik_image,
    target_dots,
    weights,
):
    """Return the geometry objective and its exact NumPy gradients."""
    ij_displacements = (
        fractional_positions[ij[:, 1]]
        - fractional_positions[ij[:, 0]]
        + ij_image
    )
    ik_displacements = (
        fractional_positions[ik[:, 1]]
        - fractional_positions[ik[:, 0]]
        + ik_image
    )
    ij_vectors = ij_displacements @ lattice
    ik_vectors = ik_displacements @ lattice
    errors = np.sum(ij_vectors * ik_vectors, axis=1) - target_dots
    value = float(np.mean(np.square(errors) * weights))

    dot_gradient = 2.0 * weights * errors / len(errors)
    ij_vector_gradient = dot_gradient[:, np.newaxis] * ik_vectors
    ik_vector_gradient = dot_gradient[:, np.newaxis] * ij_vectors
    ij_displacement_gradient = ij_vector_gradient @ lattice.T
    ik_displacement_gradient = ik_vector_gradient @ lattice.T

    position_gradient = np.zeros_like(fractional_positions)
    np.add.at(position_gradient, ij[:, 1], ij_displacement_gradient)
    np.add.at(position_gradient, ij[:, 0], -ij_displacement_gradient)
    np.add.at(position_gradient, ik[:, 1], ik_displacement_gradient)
    np.add.at(position_gradient, ik[:, 0], -ik_displacement_gradient)
    lattice_gradient = (
        ij_displacements.T @ ij_vector_gradient
        + ik_displacements.T @ ik_vector_gradient
    )
    return value, position_gradient, lattice_gradient


class NodeLatticeOptimizer:
    """Optimize topological-node positions and lattice parameters for chosen blocks."""

    def __init__(self, length_weight=1.0):
        """Store the length-versus-angle weighting for the objective."""
        self.length_weight = length_weight

    def optimize(self, topology, bbs, perms, return_result=False):
        """Optimize a topology's node positions and lattice for assigned blocks."""
        neighbors = defaultdict(list)
        images = []
        # Preserve topology images in the wrapped gauge used by optimization.
        atom_images = (
            topology.atoms.get_scaled_positions(wrap=False)
            - topology.atoms.get_scaled_positions(wrap=True)
        ).round().astype(int)
        edge_shifts = topology.edge_shifts
        for e in topology.edge_indices:
            ni, nj = topology.neighbor_list[e]
            i = ni.index
            j = nj.index
            near, far = edge_shifts[int(e)]
            image = far - near + atom_images[j] - atom_images[i]
            images.append(image)

            # Match each endpoint to the connection order used by its block.
            lengths, vectors = [], []
            for endpoint in (ni, nj):
                block = bbs[endpoint.index]
                permutation = perms[endpoint.index]
                connection = next(
                    offset for offset, neighbor in enumerate(topology.neighbor_list[endpoint.index])
                    if np.all(abs(neighbor.distance_vector + endpoint.distance_vector) < 1e-3)
                )
                lengths.append(block.lengths[permutation[connection]])
                vectors.append(block.connection_points[permutation[connection]] - block.centroid)

            edge_length = sum(lengths)
            if bbs[e] is not None:
                edge_length += 2 * bbs[e].lengths[0]

            # Set the target connection-vector length.
            vec_i, vec_j = [vector / np.linalg.norm(vector) * edge_length for vector in vectors]
            neighbors[i].append((j, image, vec_i))
            neighbors[j].append((i, -image, vec_j))

        # Ordered neighbor pairs include self-products for target lengths.
        terms = []
        for i in topology.node_indices:
            for (j, j_image, vj), (k, k_image, vk) in product(
                neighbors[i], repeat=2
            ):
                weight = 2 * self.length_weight if j == k and np.allclose(j_image, k_image) else 1.0
                terms.append(((i, j), (i, k), j_image, k_image, np.sum(vj * vk), weight))
        ij, ik, ij_image, ik_image, target_dots, weights = map(np.asarray, zip(*terms))

        # Normalize the physical scale to improve optimizer convergence.
        max_dot = np.mean(np.abs(target_dots))
        target_dots /= max_dot

        node_indices = np.asarray(topology.node_indices, dtype=int)
        slot_to_node = np.full(topology.n_slots, -1, dtype=int)
        slot_to_node[node_indices] = np.arange(len(node_indices))
        ij = slot_to_node[ij]
        ik = slot_to_node[ik]

        parameterization = SymmetryConstrainedNodeLatticeParameterization(topology)

        # Edge centers are reconstructed below. The objective evaluates all
        # node positions and the full lattice, while the optimizer sees only
        # independent orbit amplitudes and symmetry-invariant lattice strains.
        def fun_and_jac(x):
            node_parameters, lattice_parameters = parameterization.split(x)
            s = parameterization.positions(node_parameters)
            c, lattice_derivatives = parameterization.lattice.lattice_and_derivatives(lattice_parameters)
            value, position_gradient, lattice_gradient = _objective_value_and_gradient(
                s, c, ij, ik, ij_image, ik_image, target_dots, weights,
            )
            gradient = np.concatenate([
                parameterization.position_gradient(position_gradient),
                parameterization.lattice.gradient(lattice_gradient, lattice_derivatives),
            ])
            return value, gradient

        x0 = np.zeros(parameterization.n_parameters, dtype=float)
        bounds = list(parameterization.position_bounds)
        bounds.extend([(None, None)] * parameterization.n_lattice_parameters)

        result = minimize(
            x0=x0,
            fun=fun_and_jac,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
        )

        node_parameters, lattice_parameters = parameterization.split(result.x)
        c, _ = parameterization.lattice.lattice_and_derivatives(lattice_parameters)
        s = topology.atoms.get_scaled_positions(wrap=False)
        s[node_indices] = parameterization.positions(node_parameters)

        new_data = [[] for _ in range(topology.n_slots)]
        # Restore the physical length scale removed by objective normalization.
        c *= np.sqrt(max_dot)
        for e, image in zip(topology.edge_indices, images, strict=True):
            ni, nj = topology.neighbor_list[e]
            i = ni.index
            j = nj.index
            displacement = s[j] - s[i] + image
            s[e] = wrap_fractional(s[i] + 0.5 * displacement)
            d = displacement @ c
            new_data[e] += [(i, -0.5 * d), (j, 0.5 * d)]

        # Preserve connection order so existing block permutations still apply.
        for i in topology.node_indices:
            for n in topology.neighbor_list[i]:
                e = n.index
                d = n.distance_vector
                for j, en in enumerate(topology.neighbor_list[e]):
                    if np.linalg.norm(d + en.distance_vector) < 1e-4:
                        break

                _, v = new_data[e][j]
                new_data[i].append((e, -v))

        optimized_topology = topology.copy()
        optimized_topology.atoms.set_cell(c)
        optimized_topology.atoms.set_scaled_positions(wrap_fractional(s))
        optimized_topology.neighbor_list.set_data(new_data)

        if return_result:
            return optimized_topology, result
        return optimized_topology


class FixedCellPoseFitter:
    """Fit symmetry-linked rotations and translations in a fixed cell."""

    def __init__(self, framework_fitter):
        self.framework_fitter = framework_fitter

    def params(self):
        job = self.framework_fitter
        ids, frames, count = {}, {}, 0
        for orbits, compatible_orbits in (
            (job.symmetry.node_orbits, job.symmetry_compatible_node_orbits),
            (job.symmetry.edge_orbits, job.symmetry_compatible_edge_orbits),
        ):
            for orbit in orbits:
                members = [int(i) for i in orbit.members if int(i) in job.slot_offsets]
                if not members:
                    continue
                if orbit.representative not in compatible_orbits:
                    for member in members:
                        ids[member], frames[member] = count, np.eye(3)
                        count += 1
                    continue
                variable = count
                count += 1
                for member in members:
                    ids[member] = variable
                    frames[member] = (
                        np.eye(3)
                        if member == orbit.representative
                        else job.symmetry.transform(
                            job.symmetry.operation(orbit, member), job.original_cell
                        )
                    )
        return ids, frames, count

    def moves(self):
        job = self.framework_fitter
        ids, count = {}, 0
        expanded = "symmetry_operations" in job.topology.atoms.info
        anchor = job.node_indices[0] if expanded else None
        for orbits, compatible_orbits in (
            (
                job.symmetry.node_orbits if expanded else (),
                job.symmetry_compatible_node_orbits,
            ),
            (job.symmetry.edge_orbits, job.symmetry_compatible_edge_orbits),
        ):
            for orbit in orbits:
                members = [int(i) for i in orbit.members if int(i) in job.slot_offsets]
                if not members:
                    continue
                if orbit.representative in compatible_orbits:
                    if anchor not in members:
                        ids.update({member: count for member in members})
                        count += 1
                    continue
                for member in members:
                    if member != anchor:
                        ids[member] = count
                        count += 1
        return ids, count

    def fit(self, perms, centers, blocks):
        job = self.framework_fitter
        atom_pos = [
            np.round(blocks[slot].atoms.positions - blocks[slot].centroid, 10)
            for slot in job.slot_indices
        ]
        points = [
            atom_pos[i][blocks[slot].connection_point_indices]
            for i, slot in enumerate(job.slot_indices)
        ]
        rotation_ids, frames, rotation_count = self.params()
        move_ids, move_count = self.moves()
        rotation_size = 3 * rotation_count
        slots = job.slot_indices
        frame_transposes = np.asarray([frames[slot].T for slot in slots])
        rotation_variables = np.asarray([rotation_ids[slot] for slot in slots])
        rotation_frames = frame_transposes * np.asarray(
            [round(np.linalg.det(frames[slot])) for slot in slots]
        )[:, None, None]
        move_slots = np.asarray([job.slot_offsets[slot] for slot in move_ids], dtype=int)
        move_variables = np.asarray(list(move_ids.values()), dtype=int)
        left_slots = np.asarray([job.slot_offsets[c[0]] for c in job.connections])
        right_slots = np.asarray([job.slot_offsets[c[2]] for c in job.connections])
        images = np.asarray([c[4] for c in job.connections])
        point_offsets = np.cumsum([0, *[len(p) for p in points]])
        left_points = np.asarray([
            point_offsets[job.slot_offsets[left]] + perms[left][left_x]
            for left, left_x, _, _, _ in job.connections
        ])
        right_points = np.asarray([
            point_offsets[job.slot_offsets[right]] + perms[right][right_x]
            for _, _, right, right_x, _ in job.connections
        ])
        point_vectors = np.concatenate(points)
        center_positions = np.asarray(centers)

        def rotations(values):
            raw = values[:rotation_size].reshape(-1, 3)
            return Rotation.from_rotvec(
                np.einsum("nij,nj->ni", rotation_frames, raw[rotation_variables])
            )

        def residual(values):
            fitted = rotations(values)
            moves = np.zeros((len(job.slot_indices), 3))
            raw = values[rotation_size:].reshape(-1, 3)
            moves[move_slots] = np.einsum(
                "nij,nj->ni", frame_transposes[move_slots], raw[move_variables]
            )
            return (
                center_positions[right_slots]
                + moves[right_slots]
                + images
                + fitted[right_slots].apply(point_vectors[right_points])
                - center_positions[left_slots]
                - moves[left_slots]
                - fitted[left_slots].apply(point_vectors[left_points])
            ).ravel()

        result = least_squares(
            residual,
            np.zeros(rotation_size + 3 * move_count),
            max_nfev=50,
        )
        gaps = np.linalg.norm(result.fun.reshape(-1, 3), axis=1)
        fitted = rotations(result.x)
        for index, slot in enumerate(job.slot_indices):
            atom_pos[index] = fitted[index].apply(atom_pos[index])
            points[index] = atom_pos[index][blocks[slot].connection_point_indices]
        raw = result.x[rotation_size:].reshape(-1, 3)
        for slot, variable in move_ids.items():
            centers[job.slot_offsets[slot]] += frames[slot].T @ raw[variable]
        return result, gaps, atom_pos, points


class FixedCellFrameworkFitter:
    """Repair connection-site assignments and fit a framework in its source cell."""

    def __init__(self, assignment, edge_factory, *, tolerance, bond_scale):
        self.topology = assignment.topology
        self.templates = assignment.blocks
        self.blocks = assignment.placed
        self.permutations = assignment.perms
        self.provided_permutations = assignment.fixed
        self.symmetry = assignment.symmetry
        self.locator = assignment.locator
        self.pbc = assignment.pbc
        self.symmetry_compatible_node_orbits = (
            assignment.symmetry_compatible_node_orbits
        )
        self.symmetry_compatible_edge_orbits = (
            assignment.symmetry_compatible_edge_orbits
        )
        self.edge_factory = edge_factory
        self.original_cell = assignment.cell
        self.tolerance = tolerance
        self.bond_scale = bond_scale
        self.direct_edges = DirectEdgeRepair(assignment, bond_scale)
        self.direct_edge_masks = self.direct_edges.masks
        self.optimization_result = None
        self.fitted = False

    def run(self, *, fit_geometry=True):
        """Repair assignments, attempt fixed-cell fitting, and return this stage."""
        self.direct_edges.repair_if_needed()

        if fit_geometry and self._should_fit_fixed_cell():
            result = self._fit_fixed_cell()
            if result is not None:
                self.blocks, self.permutations, self.optimization_result = result
                self.fitted = True
        return self

    def edge_endpoint_offsets(self, edge_index):
        return self.direct_edges.ends(edge_index)

    def _should_fit_fixed_cell(self):
        has_edges = any(
            self.templates[int(index)] is not None
            for index in self.topology.edge_indices
        )
        eligible = bool(
            self.direct_edge_masks and not has_edges
            or has_edges and "symmetry_operations" in self.topology.atoms.info
        )
        return eligible and not self._has_pose_restricting_stabilizer()

    def _has_pose_restricting_stabilizer(self):
        """Return whether a site symmetry forbids a freely fitted rigid pose."""

        identity = np.eye(3, dtype=int)
        for orbits, compatible_orbits in (
            (self.symmetry.node_orbits, self.symmetry_compatible_node_orbits),
            (self.symmetry.edge_orbits, self.symmetry_compatible_edge_orbits),
        ):
            for orbit in orbits:
                if orbit.representative not in compatible_orbits:
                    continue
                if self.templates[orbit.representative] is None:
                    continue
                for operation_index in orbit.stabilizers:
                    rotation = np.rint(
                        self.symmetry.rotations[int(operation_index)]
                    ).astype(int)
                    if not (
                        np.array_equal(rotation, identity)
                        or np.array_equal(rotation, -identity)
                    ):
                        return True
        return False

    def _fit_fixed_cell(self):
        self.node_indices = [int(index) for index in self.topology.node_indices]
        self.edge_indices = [
            int(index) for index in self.topology.edge_indices
            if self.blocks[int(index)] is not None
        ]
        self.slot_indices = self.node_indices + self.edge_indices
        self.slot_offsets = {slot: offset for offset, slot in enumerate(self.slot_indices)}
        port_solver = PortAssignmentSolver(self)
        pose_fitter = FixedCellPoseFitter(self)
        self.connections = port_solver.connections()
        self.symmetry_compatible_edge_members = {
            int(member)
            for orbit in self.symmetry.edge_orbits
            if orbit.representative in self.symmetry_compatible_edge_orbits
            for member in orbit.members
        }
        self.fit_tolerance = min(self.tolerance, self.pbc.tolerance)

        candidate = [
            None if value is None else np.asarray(value).copy()
            for value in self.permutations
        ]
        seen = set()
        for _ in range(4):
            self._propagate_symmetry_related_permutations(candidate)
            key = tuple(np.asarray(candidate[index]).tobytes() for index in self.slot_indices)
            if key in seen:
                break
            seen.add(key)

            centers, blocks, edge_targets = self._place_candidate_blocks(candidate)
            result, gaps, atom_points, points = pose_fitter.fit(
                candidate, centers, blocks
            )
            if (
                np.max(gaps) <= self.fit_tolerance
                and port_solver.valid(candidate)
            ):
                self._place_independently_fitted_edges(
                    candidate, edge_targets, blocks, centers, atom_points, points
                )
                for offset, slot in enumerate(self.slot_indices):
                    blocks[slot].atoms.set_positions(atom_points[offset] + centers[offset])
                return blocks, candidate, OptimizeResult(fun=result.cost)
            if "symmetry_operations" not in self.topology.atoms.info:
                break
            candidate = port_solver.solve(centers, points)
            if candidate is None:
                break
        return None

    def _propagate_symmetry_related_permutations(self, candidate):
        for orbits, compatible_orbits in (
            (self.symmetry.node_orbits, self.symmetry_compatible_node_orbits),
            (self.symmetry.edge_orbits, self.symmetry_compatible_edge_orbits),
        ):
            for orbit in orbits:
                if orbit.representative not in compatible_orbits:
                    continue
                representative = candidate[orbit.representative]
                for raw_member in orbit.members:
                    member = int(raw_member)
                    candidate[member] = self.symmetry.to_member(orbit, member, representative)

    def _place_candidate_blocks(self, candidate):
        centers = self.topology.atoms.positions[self.slot_indices].copy()
        blocks = self.blocks.copy()
        for slot in self.slot_indices:
            blocks[slot] = self.templates[slot].copy()
            blocks[slot].set_centroid(self.topology.atoms.positions[slot])

        edge_targets = (
            self.edge_factory(self.topology, self.topology, candidate, self.symmetry).build()
            if self.edge_indices
            else {}
        )
        for orbits, compatible_orbits in (
            (self.symmetry.node_orbits, self.symmetry_compatible_node_orbits),
            (self.symmetry.edge_orbits, self.symmetry_compatible_edge_orbits),
        ):
            for orbit in orbits:
                members = [int(member) for member in orbit.members if int(member) in self.slot_offsets]
                if not members:
                    continue
                representative = orbit.representative if orbit.representative in members else members[0]
                for member in members:
                    if member != representative and orbit.representative in compatible_orbits:
                        blocks[member] = self.symmetry.place(
                            blocks[representative], self.symmetry.operation(orbit, member),
                            self.topology.atoms.positions[member], self.original_cell,
                        )
                    elif member in edge_targets:
                        blocks[member], _, _ = self.locator.locate_edge_with_frame(
                            self._edge_target_points(member), blocks[member], candidate[member],
                            edge_targets[member].transverse,
                        )
                    else:
                        blocks[member], _ = self.locator.locate_with_permutation(
                            self.topology.local_structure(member), blocks[member], candidate[member]
                        )
                    blocks[member].set_centroid(self.topology.atoms.positions[member])
        return centers, blocks, edge_targets

    def _edge_target_points(self, edge_index):
        return self.topology.atoms.positions[edge_index] + np.asarray(
            [neighbor.distance_vector for neighbor in self.topology.neighbor_list[edge_index]]
        )

    def _place_independently_fitted_edges(
        self, candidate, edge_targets, blocks, centers, atom_points, points
    ):
        for edge_index in self.edge_indices:
            if edge_index in self.symmetry_compatible_edge_members:
                continue
            target_points = [
                self.topology.atoms.positions[edge_index]
                + neighbor.distance_vector
                + points[self.slot_offsets[node]][candidate[node][target]]
                for neighbor, (node, target) in zip(
                    self.topology.neighbor_list[edge_index],
                    self.edge_endpoint_offsets(edge_index), strict=True,
                )
            ]
            blocks[edge_index], _, _ = self.locator.locate_edge_with_frame(
                target_points,
                self.templates[edge_index],
                candidate[edge_index],
                edge_targets[edge_index].transverse,
            )
            offset = self.slot_offsets[edge_index]
            centers[offset] = blocks[edge_index].centroid
            atom_points[offset] = blocks[edge_index].atoms.positions - centers[offset]


class Scaler:
    """Fit node positions, rotations, and the cell in one continuous solve."""

    def __init__(self, max_nfev=120):
        self.max_nfev = max_nfev

    def optimize_placement(self, placement):
        _GeometryFit(placement, self.max_nfev).run()


class _GeometryFit:
    def __init__(self, placement, max_nfev):
        self.placement = placement
        self.max_nfev = max_nfev

    def run(self):
        self._seed_cell()
        self._prepare()
        result = least_squares(
            self._residual, np.zeros(self.n_parameters), bounds=self.bounds,
            jac=self._jacobian,
            tr_solver="lsmr", tr_options={"atol": 1e-10, "btol": 1e-10},
            max_nfev=self.max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-8,
        )
        self._apply(result.x)
        self.placement.geometry_optimization_result = OptimizeResult(fun=result.cost)

    def _seed_cell(self):
        """Set an initial physical scale analytically, without a second solver."""
        p = self.placement
        self.node_indices = np.asarray(p.topology.node_indices, dtype=int)
        p.edge_shifts = p.topology.edge_shifts
        distances, targets = [], []
        for raw in p.topology.edge_indices:
            edge = int(raw)
            neighbors = p.topology.neighbor_list[edge]
            ports = p.pbc.find_matched_x_sites(
                p.source_topology, p.blocks, p.permutations, edge
            )
            distances.append(np.linalg.norm(
                neighbors[1].distance_vector - neighbors[0].distance_vector
            ))
            target = sum(np.linalg.norm(
                p.blocks[n.index].atoms.positions[x] - p.blocks[n.index].centroid
            ) for n, x in zip(neighbors, ports, strict=True))
            if p.blocks[edge] is not None:
                target += np.linalg.norm(np.diff(p.blocks[edge].connection_points, axis=0)[0])
            targets.append(target)
        factor = np.dot(distances, targets) / np.dot(distances, distances)
        topology = p.topology.copy()
        centers = topology.atoms.positions[self.node_indices] * factor
        self._update_topology(topology, centers, topology.atoms.cell.array * factor)
        p.topology = topology
        for node, center in zip(self.node_indices, centers, strict=True):
            p.blocks[int(node)].set_centroid(center)

    def _prepare(self):
        p = self.placement
        self.parameterization = SymmetryConstrainedNodeLatticeParameterization(p.topology)
        self.node_images = np.rint(
            p.topology.atoms.get_scaled_positions(wrap=False)[self.node_indices]
            - self.parameterization.initial_positions
        )
        self.node_lookup = {int(slot): i for i, slot in enumerate(self.node_indices)}
        self.relative_positions = [
            p.blocks[int(slot)].atoms.positions - p.blocks[int(slot)].centroid
            for slot in self.node_indices
        ]
        self.position_jacobian = np.zeros((len(self.node_indices), 3,
                                           self.parameterization.n_node_parameters))
        for orbit in self.parameterization.orbits:
            self.position_jacobian[orbit.member_indices, :, orbit.parameter_slice] = orbit.jacobians
        self.geometry_size = self.parameterization.n_parameters
        self.rotation_maps = []
        self.shared_node_orbits = []
        self.n_parameters = self.geometry_size
        lower, upper = [], []
        for lo, hi in self.parameterization.position_bounds:
            lower.append(max(lo, -.1))
            upper.append(min(hi, .1))
        # Bounds keep node images and the initial orientation branch fixed.
        lower.extend([-.25] * self.parameterization.n_lattice_parameters)
        upper.extend([.25] * self.parameterization.n_lattice_parameters)
        cell = np.asarray(p.topology.atoms.cell)
        for orbit in p.symmetry.node_orbits:
            compatible = orbit.representative in p.symmetry_compatible_node_orbits
            if compatible:
                # Rotation vectors are axial vectors, including under mirrors.
                transforms = [p.symmetry.transform(int(op), cell) for op in orbit.stabilizers]
                constraints = [np.linalg.det(t) * t.T - np.eye(3) for t in transforms]
                basis = Symmetry.null_space(np.concatenate(constraints), 3)
                section = slice(self.n_parameters, self.n_parameters + basis.shape[1])
                self.n_parameters = section.stop
                lower.extend([-np.pi/4] * basis.shape[1])
                upper.extend([np.pi/4] * basis.shape[1])
                for raw in orbit.members:
                    member = int(raw)
                    transform = p.symmetry.transform(p.symmetry.operation(orbit, member), cell)
                    mapping = round(np.linalg.det(transform)) * transform.T @ basis
                    self.rotation_maps.append((self.node_lookup[member], section, mapping))
                self.shared_node_orbits.append(orbit)
            else:
                for raw in orbit.members:
                    section = slice(self.n_parameters, self.n_parameters + 3)
                    self.n_parameters = section.stop
                    self.rotation_maps.append((self.node_lookup[int(raw)], section, np.eye(3)))
                    lower.extend([-np.pi/4] * 3)
                    upper.extend([np.pi/4] * 3)
        self.bounds = np.asarray(lower), np.asarray(upper)
        left, right, shifts, left_ports, right_ports, spans, direct = [], [], [], [], [], [], []
        for raw in p.topology.edge_indices:
            edge = int(raw)
            ni, nj = p.topology.neighbor_list[edge]
            xi, xj = p.pbc.find_matched_x_sites(
                p.source_topology, p.blocks, p.permutations, edge
            )
            i, j = self.node_lookup[ni.index], self.node_lookup[nj.index]
            left.append(i)
            right.append(j)
            left_ports.append(self.relative_positions[i][xi])
            right_ports.append(self.relative_positions[j][xj])
            shifts.append(p.edge_shifts[edge][1] - p.edge_shifts[edge][0])
            linker = p.blocks[edge]
            direct.append(linker is None)
            spans.append(0.0 if linker is None else float(np.linalg.norm(
                np.diff(linker.connection_points, axis=0)[0]
            )))
        self.left, self.right = np.asarray(left), np.asarray(right)
        self.shifts = (np.asarray(shifts) + self.node_images[self.right]
                       - self.node_images[self.left])
        self.left_ports, self.right_ports = np.asarray(left_ports), np.asarray(right_ports)
        self.spans, self.direct = np.asarray(spans), np.asarray(direct)

    def _state(self, values):
        node_values, lattice_values = self.parameterization.split(values[:self.geometry_size])
        fractional = self.parameterization.positions(node_values)
        cell, derivatives = self.parameterization.lattice.lattice_and_derivatives(lattice_values)
        vectors = np.zeros((len(self.node_indices), 3))
        for node, section, mapping in self.rotation_maps:
            vectors[node] = mapping @ values[section]
        return fractional, cell, Rotation.from_rotvec(vectors), derivatives

    def _residual(self, values):
        fractional, cell, rotations, _ = self._state(values)
        vectors = (
            (fractional[self.right] - fractional[self.left] + self.shifts) @ cell
            + rotations[self.right].apply(self.right_ports)
            - rotations[self.left].apply(self.left_ports)
        )
        # Each linker contributes its actual port-span error. Direct node-node
        # connections must coincide as vectors, not merely have a target length.
        # A weak pose prior resolves free torsions without deforming the blocks.
        return np.concatenate([
            np.linalg.norm(vectors[~self.direct], axis=1) - self.spans[~self.direct],
            vectors[self.direct].ravel(),
            1e-3 * values,
        ])

    @staticmethod
    def _skew(vectors):
        vectors = np.asarray(vectors)
        matrices = np.zeros((*vectors.shape[:-1], 3, 3))
        matrices[..., 0, 1] = -vectors[..., 2]
        matrices[..., 0, 2] = vectors[..., 1]
        matrices[..., 1, 0] = vectors[..., 2]
        matrices[..., 1, 2] = -vectors[..., 0]
        matrices[..., 2, 0] = -vectors[..., 1]
        matrices[..., 2, 1] = vectors[..., 0]
        return matrices

    def _jacobian(self, values):
        fractional, cell, rotations, cell_derivatives = self._state(values)
        delta = fractional[self.right] - fractional[self.left] + self.shifts
        vectors = (delta @ cell + rotations[self.right].apply(self.right_ports)
                   - rotations[self.left].apply(self.left_ports))
        jac = np.zeros((len(self.left), 3, self.n_parameters))
        npos = self.parameterization.n_node_parameters
        jac[:, :, :npos] = np.einsum(
            'eik,ij->ejk', self.position_jacobian[self.right]-self.position_jacobian[self.left], cell)
        jac[:, :, npos:self.geometry_size] = np.einsum('ei,kij->ejk', delta, cell_derivatives)
        rotvecs, matrices = rotations.as_rotvec(), rotations.as_matrix()
        for node, section, mapping in self.rotation_maps:
            if section.start == section.stop:
                continue
            vector = rotvecs[node]
            theta = np.linalg.norm(vector)
            skew = self._skew(vector)
            if theta < 1e-5:
                a, b = .5-theta**2/24, 1/6-theta**2/120
            else:
                a, b = (1-np.cos(theta))/theta**2, (theta-np.sin(theta))/theta**3
            right_jacobian = np.eye(3)-a*skew+b*(skew@skew)
            for slots, ports, sign in [(self.left, self.left_ports, -1),
                                       (self.right, self.right_ports, 1)]:
                indices = np.flatnonzero(slots == node)
                deriv = -np.einsum('ij,ejk,kl->eil', matrices[node],
                                   self._skew(ports[indices]), right_jacobian @ mapping)
                jac[indices, :, section] += sign*deriv
        lengths = np.linalg.norm(vectors, axis=1)
        unit = vectors / np.maximum(lengths[:, None], 1e-15)
        return np.vstack([
            np.einsum('ei,eik->ek', unit[~self.direct], jac[~self.direct]),
            jac[self.direct].reshape(-1, self.n_parameters),
            1e-3*np.eye(self.n_parameters),
        ])

    def _apply(self, values):
        p = self.placement
        fractional, cell, rotations, _ = self._state(values)
        centers = (fractional + self.node_images) @ cell
        for i, slot in enumerate(self.node_indices):
            p.blocks[int(slot)].atoms.set_positions(
                rotations[i].apply(self.relative_positions[i]) + centers[i]
            )
        self._update_topology(p.topology, centers, cell)
        p.edge_shifts = p.topology.edge_shifts
        for orbit in self.shared_node_orbits:
            diagnostic = p.symmetry.diagnose(
                orbit, p.blocks[orbit.representative],
                p.permutations[orbit.representative], cell, p.symmetry_tolerance,
            )
            if not diagnostic["compatible"]:
                raise SymmetryCompatibilityError(
                    "final_stabilizer_incompatible", kind="node",
                    representative=orbit.representative, diagnostic=diagnostic,
                )
            diagnostic.update(kind="node", symmetry_propagated=True)
            p.symmetry_diagnostics["nodes"].append(diagnostic)
        p.rmsds = np.array([
            np.sqrt(np.mean(np.sum((
                p.topology.local_structure(int(node)).atoms.positions
                - p.blocks[int(node)].local_structure().atoms.positions[p.permutations[int(node)]]
            ) ** 2, axis=1))) for node in self.node_indices
        ])

    def _update_topology(self, topology, centers, cell):
        """Move graph centers without changing any declared periodic images."""
        original = self.placement.topology
        positions = np.asarray(topology.atoms.positions).copy()
        positions[self.node_indices] = centers
        data = [[] for _ in range(topology.n_slots)]
        for raw in topology.edge_indices:
            edge = int(raw)
            ni, nj = original.neighbor_list[edge]
            shifts = self.placement.edge_shifts[edge]
            ends = positions[[ni.index, nj.index]] + shifts @ cell
            positions[edge] = ends.mean(axis=0)
            data[edge] = [(n.index, end - positions[edge])
                          for n, end in zip((ni, nj), ends, strict=True)]
        for raw in self.node_indices:
            node = int(raw)
            for neighbor in original.neighbor_list[node]:
                edge = neighbor.index
                matches = [j for j, back in enumerate(original.neighbor_list[edge])
                           if back.index == node and np.allclose(
                               back.distance_vector, -neighbor.distance_vector,
                               atol=1e-7, rtol=0)]
                if len(matches) != 1:
                    raise ValueError("Connection-gap fitting requires unique periodic incidences.")
                data[node].append((edge, -data[edge][matches[0]][1]))
        topology.atoms.set_cell(cell)
        topology.atoms.set_positions(positions)
        topology.neighbor_list.set_data(data)
