"""Symmetry-constrained optimization of node positions and lattice parameters."""
from collections import defaultdict
from itertools import product

import numpy as np
import scipy as sp

from mofbb.compose.geometry_parameterization import (
    SymmetryConstrainedNodeLatticeParameterization,
)
from mofbb.compose.utils import wrap_fractional


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
        # Get pairs of bond indices and images (periodic boundary) and
        pairs = []
        images = []
        # Preserve topology images in the wrapped gauge used by optimization.
        atom_images = (
            topology.atoms.get_scaled_positions(wrap=False)
            - topology.atoms.get_scaled_positions(wrap=True)
        ).round().astype(int)
        edge_shifts = topology.edge_shifts
        for e in topology.edge_indices:
            # ni: neigbor with index i.
            ni, nj = topology.neighbor_list[e]

            i = ni.index
            j = nj.index

            # Save index pair.
            pairs.append([i, j])

            near, far = edge_shifts[int(e)]
            images.append(far - near + atom_images[j] - atom_images[i])

        # Type casting to np.array.
        pairs = np.array(pairs)
        images = np.around(images)

        # Calculate target norms and vectors for angles.
        # ij_vectors: list of vectors node i to j with building block size.
        ij_vectors = []
        ji_vectors = []
        for e in topology.edge_indices:
            # ni: neigbor with index i.
            ni, nj = topology.neighbor_list[e]

            i = ni.index
            j = nj.index

            # Find connection point index.
            for ci, n in enumerate(topology.neighbor_list[i]):
                zero_sum = np.abs(n.distance_vector + ni.distance_vector)
                if (zero_sum < 1e-3).all():
                    # ci saved.
                    break

            for cj, n in enumerate(topology.neighbor_list[j]):
                zero_sum = np.abs(n.distance_vector + nj.distance_vector)
                if (zero_sum < 1e-3).all():
                    # cj saved.
                    break

            # Get node bb length to the connection point.
            # cp: connection point.
            bb = bbs[i]
            p = perms[i]
            len_i = bb.lengths[p][ci]
            vec_i = bb.connection_points[p][ci] - bb.centroid

            bb = bbs[j]
            p = perms[j]
            len_j = bb.lengths[p][cj]
            vec_j = bb.connection_points[p][cj] - bb.centroid

            edge_length = len_i + len_j
            if bbs[e] is not None:
                edge_length += 2 * bbs[e].lengths[0]

            # Set the target connection-vector length.
            vec_i = vec_i / np.linalg.norm(vec_i) * edge_length
            vec_j = vec_j / np.linalg.norm(vec_j) * edge_length

            ij_vectors.append(vec_i)
            ji_vectors.append(vec_j)

        # Cast to numpy array.
        ij_vectors = np.array(ij_vectors)
        ji_vectors = np.array(ji_vectors)

        # Get angle triples.
        # Triples are used for tensor operations during optimization.

        # New data view of pairs and images for estimation of triples.
        data_view = defaultdict(list)
        for (i, j), image in zip(pairs, images):
            data_view[i].append((j, image))
            data_view[j].append((i, -image))

        # Triples for the calculatation of dots between r_{ij} and r_{ik}.
        ij = []
        ik = []

        ij_image = []
        ik_image = []

        # Weights for objective function.
        weights = []

        for i in topology.node_indices:
            neigbors = data_view[i]
            for (j, j_image), (k, k_image) in product(neigbors, repeat=2):
                ij.append([i, j])
                ik.append([i, k])

                ij_image.append(j_image)
                ik_image.append(k_image)

                if (j == k) and np.allclose(j_image, k_image):
                    # 2 for count collection for dot product of same edges.
                    weights.append(2 * self.length_weight)
                else:
                    weights.append(1.0)

        # Type cast.
        ij = np.array(ij)
        ik = np.array(ik)

        ij_image = np.array(ij_image)
        ik_image = np.array(ik_image)

        weights = np.array(weights)

        # Calculate target angles.
        # Similar method to above loops.
        vectors_view = defaultdict(list)
        for (i, j), v_ij, v_ji in zip(pairs, ij_vectors, ji_vectors):
            vectors_view[i].append(v_ij)
            vectors_view[j].append(v_ji)

        # Now, i represents node i (center). j and k are represent indices of
        # connection points from i.
        target_dots = []
        target_ij_vec = []
        target_ik_vec = []
        for i in topology.node_indices:
            # Get all connection point vectors of node i.
            vectors = vectors_view[i]
            # Product includes self dot product (vj == vk).
            for vj, vk in product(vectors, repeat=2):
                target_ij_vec.append(vj)
                target_ik_vec.append(vk)

        target_ij_vec = np.array(target_ij_vec)
        target_ik_vec = np.array(target_ik_vec)

        target_dots = np.sum(target_ij_vec * target_ik_vec, axis=-1)

        # Normalize target dots. This enhances the optimization convegences.
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
            c, lattice_derivatives = (
                parameterization.lattice.lattice_and_derivatives(lattice_parameters)
            )
            value, position_gradient, lattice_gradient = (
                _objective_value_and_gradient(
                    s,
                    c,
                    ij,
                    ik,
                    ij_image,
                    ik_image,
                    target_dots,
                    weights,
                )
            )
            gradient = np.concatenate(
                [
                    parameterization.pos_grad(position_gradient),
                    parameterization.lattice.grad(
                        lattice_gradient,
                        lattice_derivatives,
                    ),
                ]
            )
            return value, gradient

        x0 = np.zeros(parameterization.n_parameters, dtype=float)
        bounds = list(parameterization.position_bounds)
        bounds.extend([(None, None)] * parameterization.n_lattice_parameters)

        result = sp.optimize.minimize(
            x0=x0,
            fun=fun_and_jac,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "disp": False},
        )

        node_parameters, lattice_parameters = parameterization.split(result.x)
        c, _ = parameterization.lattice.lattice_and_derivatives(lattice_parameters)
        s = topology.atoms.get_scaled_positions(wrap=False)
        s[node_indices] = parameterization.positions(node_parameters)

        # Update neigbors list in topology.
        new_data = [[] for _ in range(topology.n_slots)]
        # Restore the physical length scale removed by objective normalization.
        c *= np.sqrt(max_dot)
        # Transform to Cartesian coordinates.
        r = s @ c
        invc = np.linalg.inv(c)
        for e, image in zip(topology.edge_indices, images, strict=True):
            ni, nj = topology.neighbor_list[e]

            i = ni.index
            j = nj.index

            # Calculate new edge center.
            ri = r[i]
            rj = r[j]

            d = rj - ri + np.dot(image, c)

            # Select center position wrapped by unit cell.
            rc = ri + 0.5 * d
            sc = np.dot(rc, invc)
            sc = wrap_fractional(sc)
            rc = np.dot(sc, c)
            r[e] = rc

            # Save in proper order.
            new_data[e] += [(i, -0.5 * d), (j, 0.5 * d)]

        # Should change this stupidly nested loop.
        # The new neigbor list is updated with same order of original neigbor
        # list. Then we can use the permutation information for new location
        # after node-position and lattice optimization.
        for i in topology.node_indices:
            # Same order loop. Note that topology is the original.
            for n in topology.neighbor_list[i]:
                e = n.index
                d = n.distance_vector
                # Find cross reference.
                for j, en in enumerate(topology.neighbor_list[e]):
                    if np.linalg.norm(d + en.distance_vector) < 1e-4:
                        # j and en saved.
                        break

                _, v = new_data[e][j]
                new_data[i].append((e, -v))

        # Construct the optimized topology.
        s = np.dot(r, invc)
        s = wrap_fractional(s)
        r = np.dot(s, c)

        optimized_topology = topology.copy()
        optimized_topology.atoms.set_positions(r)
        optimized_topology.atoms.set_cell(c)
        optimized_topology.neighbor_list.set_data(new_data)

        if return_result:
            return optimized_topology, result
        return optimized_topology
