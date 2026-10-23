"""Joint port-assignment, rigid-pose, and lattice optimization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.groups import SpaceGroup
from scipy.optimize import OptimizeResult, least_squares
from scipy.spatial.transform import Rotation
from scipy.special import exprel

from mofbb.compose.placement.errors import SymmetryCompatibilityError
from mofbb.compose.placement.ports import (
    PortAssignmentSolver,
    connection_residuals,
)


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
    member_indices: np.ndarray
    operation_indices: np.ndarray
    site_symmetry_operation_indices: np.ndarray
    basis: np.ndarray
    base_positions: np.ndarray
    jacobians: np.ndarray
    parameter_slice: slice
    bounds: tuple[tuple[float, float], ...]

    @property
    def dimension(self):
        return self.basis.shape[1]


class OptParams:
    """Define all node-pose and cell variables, bounds, and state expansion."""

    def __init__(self, topology, map_tol=2.0e-5, *, compatible_node_orbits=()):
        self.map_tol = map_tol
        self.node_idx = np.asarray(topology.node_idx, dtype=int)
        self.node_lookup = {slot: i for i, slot in enumerate(self.node_idx)}
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
        self.initial_positions = np.asarray(positions[self.node_idx], dtype=float)
        self.node_images = np.rint(
            topology.atoms.get_scaled_positions(wrap=False)[self.node_idx] - self.initial_positions
        )
        self.tags = np.asarray(topology.node_types, dtype=int)[self.node_idx]
        self.operation_mappings = self.map_nodes()
        self.identity_operation_index = self.find_identity()
        self.orbits, self.position_bounds = self.build_orbits()
        self.n_node_parameters = sum(orbit.dimension for orbit in self.orbits)
        self.base_positions = np.empty_like(self.initial_positions)
        self.position_jacobian = np.zeros((len(self.node_idx), 3, self.n_node_parameters))
        for orbit in self.orbits:
            self.base_positions[orbit.member_indices] = orbit.base_positions
            self.position_jacobian[orbit.member_indices, :, orbit.parameter_slice] = orbit.jacobians
        self.initial_lattice = np.asarray(topology.atoms.cell, dtype=float).copy()
        if not np.all(np.isfinite(self.initial_lattice)):
            raise ValueError("The initial topology lattice is non-finite.")
        if abs(np.linalg.det(self.initial_lattice)) < np.finfo(float).eps:
            raise ValueError("The initial topology lattice is singular.")
        self.cartesian_rotations = self._cartesian_rotations()
        self.lattice_basis = self._strain_basis()
        self.n_lattice_parameters = len(self.lattice_basis)
        self.node_slice = slice(0, self.n_node_parameters)
        self.lattice_slice = slice(self.node_slice.stop, self.node_slice.stop + self.n_lattice_parameters)
        self.n_parameters = self.lattice_slice.stop
        self.rotation_maps = []
        lower = [max(lo, -.1) for lo, _ in self.position_bounds] + [-.25] * self.n_lattice_parameters
        upper = [min(hi, .1) for _, hi in self.position_bounds] + [.25] * self.n_lattice_parameters
        for nodes, axes in self._rotation_groups(compatible_node_orbits):
            count = axes.shape[-1]
            section = slice(self.n_parameters, self.n_parameters + count)
            self.n_parameters = section.stop
            lower.extend([-np.pi/4] * count)
            upper.extend([np.pi/4] * count)
            self.rotation_maps.extend(
                (node, section, mapping) for node, mapping in zip(nodes, axes, strict=True)
            )
        self.n_rotation_parameters = self.n_parameters - self.lattice_slice.stop
        self.bounds = np.asarray(lower), np.asarray(upper)

    def _rotation_groups(self, compatible_node_orbits):
        transforms = (np.linalg.inv(self.initial_lattice)
                      @ self.rotations.swapaxes(1, 2) @ self.initial_lattice)
        for orbit in self.orbits:
            representative = self.node_idx[orbit.member_indices[0]]
            if representative in compatible_node_orbits:
                # Rotation vectors are axial vectors, including under mirrors.
                stabilizers = transforms[orbit.site_symmetry_operation_indices]
                axial = np.linalg.det(stabilizers)[:, None, None] * stabilizers.swapaxes(1, 2)
                basis = Symmetry.null_space(axial - np.eye(3), 3)
                members = transforms[orbit.operation_indices]
                axes = np.rint(np.linalg.det(members))[:, None, None] * members.swapaxes(1, 2) @ basis
                yield orbit.member_indices, axes
            else:
                for node in orbit.member_indices:
                    yield (node,), np.eye(3)[None]

    def state(self, values):
        """Expand a full vector into unwrapped node positions, cell, and rotations."""
        values = np.asarray(values, dtype=float)
        fractional = self.positions(values[self.node_slice]) + self.node_images
        cell, derivatives = self.lattice_and_derivatives(values[self.lattice_slice])
        vectors = np.zeros((len(self.node_idx), 3))
        for node, section, mapping in self.rotation_maps:
            vectors[node] = mapping @ values[section]
        return fractional, cell, Rotation.from_rotvec(vectors), derivatives

    def _cartesian_rotations(self, lattice_tol=1.0e-10):
        metric = self.initial_lattice @ self.initial_lattice.T
        scale = max(1.0, np.max(np.abs(metric)))
        rotations = np.array(list({
            tuple(np.rint(rotation).astype(int).ravel()): rotation
            for rotation in self.rotations
        }.values()))
        transposed = rotations.swapaxes(1, 2)
        residuals = np.max(abs(transposed @ metric @ rotations - metric), axis=(1, 2)) / scale
        invalid = np.flatnonzero(residuals > lattice_tol)
        if len(invalid):
            index = invalid[0]
            raise ValueError(
                f"The initial lattice is incompatible with declared point operation {index}: "
                f"relative metric residual {residuals[index]:.3e}."
            )
        return np.linalg.inv(self.initial_lattice) @ transposed @ self.initial_lattice

    def _strain_basis(self):
        symmetric = np.zeros((6, 3, 3), dtype=float)
        symmetric[np.arange(3), np.arange(3), np.arange(3)] = 1.0
        value = 1.0 / np.sqrt(2.0)
        symmetric[3, 0, 1] = symmetric[3, 1, 0] = value
        symmetric[4, 0, 2] = symmetric[4, 2, 0] = value
        symmetric[5, 1, 2] = symmetric[5, 2, 1] = value

        rotations = self.cartesian_rotations[:, None]
        changes = rotations @ symmetric @ rotations.swapaxes(-1, -2) - symmetric
        coefficients = Symmetry.null_space(changes.reshape(-1, 6, 9).swapaxes(1, 2), 6)
        return np.einsum("ak,aij->kij", coefficients, symmetric)

    def lattice_and_derivatives(self, parameters):
        """Return the strained lattice and its exact parameter derivatives."""

        parameters = np.asarray(parameters, dtype=float)
        strain = np.einsum("k,kij->ij", parameters, self.lattice_basis)
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
        rotated = vectors.T @ self.lattice_basis @ vectors
        frechet = vectors @ (rotated * divided) @ vectors.T
        derivatives = self.initial_lattice @ frechet
        if not np.all(np.isfinite(derivatives)):
            raise ValueError(
                "The symmetry-constrained lattice derivatives became non-finite."
            )
        return lattice, derivatives

    def map_nodes(self):
        mappings = []
        node_count = len(self.node_idx)
        for operation_index, (rotation, translation) in enumerate(
            zip(self.rotations, self.translations)
        ):
            transformed = self.initial_positions @ rotation.T + translation
            differences = Symmetry.diff(
                transformed[:, None, :], self.initial_positions[None, :, :]
            )
            matches = np.all(np.abs(differences) <= self.map_tol, axis=2)
            invalid = np.flatnonzero(matches.sum(axis=1) != 1)
            if len(invalid):
                slot = invalid[0]
                targets = self.node_idx[np.flatnonzero(matches[slot])].tolist()
                raise ValueError(
                    f"Declared symmetry operation {operation_index} must map node slot "
                    f"{self.node_idx[slot]} uniquely within fractional tolerance "
                    f"{self.map_tol:.1e}; matched slots {targets}."
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
        return indices[0]

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
        members = members[np.argsort(self.node_idx[members])]
        representative = members[0]
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
            member_indices=members,
            operation_indices=operation_indices,
            site_symmetry_operation_indices=stabilizer,
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
        return tuple((-value, value) for value in upper)

    def positions(self, node_parameters):
        """Expand orbit amplitudes into all node positions."""
        return self.base_positions + np.einsum("mrd,d->mr", self.position_jacobian, node_parameters)


class Scaler:
    """Alternate port assignments with a joint fit of node poses and the cell."""

    def __init__(self, max_nfev=120):
        self.max_nfev = max_nfev

    def optimize(self, placement):
        """Keep improving feasible assignments and continuous geometry together."""
        PortAssignmentSolver.initialize(placement)
        initial_diagnostics = deepcopy(placement.symmetry_diagnostics)
        fields = (
            "topology", "blocks", "perms", "symmetry_diagnostics",
            "edge_targets", "edge_frame_diagnostics", "edge_shifts", "rmsds",
            "geometry_optimization_result", "permutation_optimization_steps",
        )
        best, best_score, best_feasible, seen = None, np.inf, False, set()
        for step in range(4):
            key = tuple(None if perm is None else tuple(perm) for perm in placement.perms)
            if key in seen:
                break
            placement.symmetry_diagnostics = deepcopy(initial_diagnostics)
            try:
                GeomOpt(placement, self.max_nfev).solve(seed=step == 0)
                placement.run()
            except ValueError:
                if best is None:
                    raise
                break
            solver = PortAssignmentSolver(placement)
            blocks = [placement.blocks[slot] for slot in solver.slot_indices]
            centers = np.asarray([block.centroid for block in blocks])
            points = [block.port_pos - block.centroid for block in blocks]
            gaps = solver.residuals(centers, points, placement.perms)
            score = solver.loss(centers, points, placement.perms)
            feasible = bool(np.all(np.linalg.norm(gaps, axis=1) <= placement.port_tol))
            if best is not None:
                if best_feasible and not feasible:
                    break
                if feasible == best_feasible and score >= best_score - 1e-12 * max(1.0, best_score):
                    break
            seen.add(tuple(None if perm is None else tuple(perm) for perm in placement.perms))
            placement.permutation_optimization_steps = step
            best_score = score
            best_feasible = feasible
            best = deepcopy({field: getattr(placement, field) for field in fields})
            if score < 1e-12:
                break
            candidate = solver.solve(centers, points)
            if candidate is None:
                break
            candidate_score = solver.loss(centers, points, candidate)
            if candidate_score >= score - 1e-12 * max(1.0, score):
                break
            placement.perms = candidate
        for field, value in best.items():
            setattr(placement, field, value)


class GeomOpt:
    """Fit the joint loss using the complete variables defined by OptParams."""

    def __init__(self, placement, max_nfev):
        self.placement = placement
        self.max_nfev = max_nfev

    def solve(self, *, seed=True):
        self.placement.edge_shifts = self.placement.topology.edge_shifts
        if seed:
            self._init_scale()
        self._setup_opt()
        result = least_squares(
            self._calc_res, np.zeros(self.parameters.n_parameters), bounds=self.parameters.bounds,
            jac=self._calc_jacobian,
            tr_solver="lsmr", tr_options={"atol": 1e-10, "btol": 1e-10},
            max_nfev=self.max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-8,
        )
        self._update_geom(result.x)
        self.placement.geometry_optimization_result = OptimizeResult(fun=2 * result.cost)

    def _init_scale(self):
        """Set an initial physical scale analytically, without a second solver."""
        p = self.placement
        distances, targets = [], []
        for edge in p.topology.edge_idx:
            neighbors = p.topology.neighbor_list[edge]
            ports = p.pbc.find_matched_x_sites(
                p.source_topology, p.blocks, p.perms, edge
            )
            distances.append(np.linalg.norm(
                neighbors[1].distance_vector - neighbors[0].distance_vector
            ))
            target = sum(np.linalg.norm(
                p.blocks[n.index].atoms.positions[x] - p.blocks[n.index].centroid
            ) for n, x in zip(neighbors, ports, strict=True))
            if p.blocks[edge] is not None:
                target += np.linalg.norm(np.diff(p.blocks[edge].port_pos, axis=0)[0])
            targets.append(target)
        factor = np.dot(distances, targets) / np.dot(distances, distances)
        topology = p.topology.copy()
        centers = topology.atoms.positions[topology.node_idx] * factor
        self._update_topo(topology, centers, topology.atoms.cell.array * factor)
        p.topology = topology
        for node, center in zip(topology.node_idx, centers, strict=True):
            p.blocks[node].set_centroid(center)

    def _setup_opt(self):
        p = self.placement
        self.parameters = OptParams(
            p.topology, compatible_node_orbits=p.symmetry_compatible_node_orbits
        )
        self.relative_positions = [
            p.blocks[slot].atoms.positions - p.blocks[slot].centroid
            for slot in self.parameters.node_idx
        ]
        solver = PortAssignmentSolver(p)
        self.left, self.right = np.array([
            (self.parameters.node_lookup[left], self.parameters.node_lookup[right])
            for left, _, right, _, _ in solver.edges
        ]).T
        self.left_ports, self.right_ports = np.array([
            [p.blocks[node].port_pos[p.perms[node][target]] - p.blocks[node].centroid
             for node, target in ((left, x_i), (right, x_j))]
            for left, x_i, right, x_j, _ in solver.edges
        ]).transpose(1, 0, 2)
        self.shifts = solver.shifts
        self.spans, self.direct = solver.spans, solver.direct

    def _calc_res(self, values):
        fractional, cell, rotations, _ = self.parameters.state(values)
        vectors = (
            (fractional[self.right] - fractional[self.left] + self.shifts) @ cell
            + rotations[self.right].apply(self.right_ports)
            - rotations[self.left].apply(self.left_ports)
        )
        return connection_residuals(vectors, self.spans, self.direct)

    @staticmethod
    def _skew(vectors):
        """Return matrices that apply cross products with the input vectors."""
        return np.cross(np.eye(3), np.asarray(vectors)[..., None, :])

    def _calc_jacobian(self, values):
        parameters = self.parameters
        fractional, cell, rotations, cell_derivatives = parameters.state(values)
        delta = fractional[self.right] - fractional[self.left] + self.shifts
        vectors = (delta @ cell + rotations[self.right].apply(self.right_ports)
                   - rotations[self.left].apply(self.left_ports))
        jac = np.zeros((len(self.left), 3, parameters.n_parameters))
        jac[:, :, parameters.node_slice] = np.einsum(
            'eik,ij->ejk', parameters.position_jacobian[self.right]
            - parameters.position_jacobian[self.left], cell)
        jac[:, :, parameters.lattice_slice] = np.einsum('ei,kij->ejk', delta, cell_derivatives)
        # Compute rotation tangents in a batch, then update only their parameter columns.
        rotvecs, matrices = rotations.as_rotvec(), rotations.as_matrix()
        theta = np.linalg.norm(rotvecs, axis=1)
        safe_theta = np.maximum(theta, 1e-5)
        a = np.where(theta < 1e-5, .5-theta**2/24, (1-np.cos(theta))/safe_theta**2)
        b = np.where(theta < 1e-5, 1/6-theta**2/120, (theta-np.sin(theta))/safe_theta**3)
        skew = self._skew(rotvecs)
        right_jacobian = np.eye(3) - a[:, None, None]*skew + b[:, None, None]*(skew@skew)
        for node, section, mapping in parameters.rotation_maps:
            if section.start == section.stop:
                continue
            for slots, ports, sign in [(self.left, self.left_ports, 1), (self.right, self.right_ports, -1)]:
                indices = np.flatnonzero(slots == node)
                jac[indices, :, section] += sign * (
                    matrices[node] @ self._skew(ports[indices]) @ (right_jacobian[node] @ mapping)
                )
        lengths = np.linalg.norm(vectors, axis=1)
        unit = vectors / np.maximum(lengths[:, None], 1e-15)
        return np.vstack([
            np.einsum('ei,eik->ek', unit[~self.direct], jac[~self.direct]) / np.sqrt(2.0),
            jac[self.direct].reshape(-1, parameters.n_parameters),
        ])

    def _update_geom(self, values):
        p = self.placement
        fractional, cell, rotations, _ = self.parameters.state(values)
        centers = fractional @ cell
        for i, slot in enumerate(self.parameters.node_idx):
            p.blocks[slot].atoms.set_positions(
                rotations[i].apply(self.relative_positions[i]) + centers[i]
            )
        self._update_topo(p.topology, centers, cell)
        p.edge_shifts = p.topology.edge_shifts
        for orbit in p.symmetry.node_orbits:
            if orbit.representative not in p.symmetry_compatible_node_orbits:
                continue
            diagnostic = p.symmetry.diagnose(
                orbit, p.blocks[orbit.representative],
                p.perms[orbit.representative], cell, p.sym_tol,
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
                p.topology.local_structure(node).atoms.positions
                - p.blocks[node].local_structure().atoms.positions[p.perms[node]]
            ) ** 2, axis=1))) for node in self.parameters.node_idx
        ])

    def _update_topo(self, topology, centers, cell):
        """Move graph centers without changing any declared periodic images."""
        original = self.placement.topology
        positions = np.asarray(topology.atoms.positions).copy()
        positions[original.node_idx] = centers
        data = [[] for _ in range(topology.n_slots)]
        for edge in topology.edge_idx:
            ni, nj = original.neighbor_list[edge]
            shifts = self.placement.edge_shifts[edge]
            ends = positions[[ni.index, nj.index]] + shifts @ cell
            positions[edge] = ends.mean(axis=0)
            data[edge] = [(n.index, end - positions[edge])
                          for n, end in zip((ni, nj), ends, strict=True)]
        for node in original.node_idx:
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
