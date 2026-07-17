"""Regression tests for crystallographically reduced topology scaling."""

from pathlib import Path
from types import SimpleNamespace

from ase import Atoms
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.groups import SpaceGroup
import scipy.linalg

from mofbb.compose import BuildingBlock, Scaler, Topology
from mofbb.compose.scaler import _objective_value_and_gradient
from mofbb.compose.symmetry_scaling import SymmetryReducedParameterization


def _periodic_difference(left, right):
    difference = np.asarray(left) - np.asarray(right)
    return difference - np.rint(difference)


def _make_p4mm_node_topology():
    lattice = Lattice.tetragonal(8.0, 12.0)
    structure = Structure.from_spacegroup(
        "P4mm",
        lattice,
        ["C", "C"],
        [[0.0, 0.0, 0.25], [0.123, 0.234, 0.345]],
        site_properties={"type": [0, 1]},
    )
    tags = np.asarray(structure.site_properties["type"], dtype=int)
    atoms = Atoms(
        symbols=[site.specie.symbol for site in structure],
        scaled_positions=structure.frac_coords,
        cell=structure.lattice.matrix,
        tags=tags,
        pbc=True,
    )
    node_indices = np.arange(len(atoms), dtype=int)
    return SimpleNamespace(
        atoms=atoms,
        spacegroup="P4mm",
        node_indices=node_indices,
        node_types=tags,
    )


def _write_p4mm_cycle_topology(path: Path) -> Path:
    path.write_text(
        """CRYSTAL
  NAME p4mm-cycle
  GROUP P4mm
  CELL 10 10 10 90 90 90
  NODE 1 2 0.25 0 0.5
  EDGE 0.25 0 0.5 0 0.25 0.5
END
""",
        encoding="utf-8",
    )
    return path


def _write_linear_block(path: Path) -> Path:
    path.write_text(
        """4
linear
C -0.5 0 0
C  0.5 0 0
X -1.0 0 0
X  1.0 0 0
0 1 S
0 2 S
1 3 S
""",
        encoding="utf-8",
    )
    return path


def test_symmetry_orbits_reconstruct_and_operations_are_bijective() -> None:
    topology = _make_p4mm_node_topology()
    parameterization = SymmetryReducedParameterization(topology)
    initial = topology.atoms.get_scaled_positions()[topology.node_indices]
    zero_parameters = np.zeros(parameterization.n_node_parameters)
    reconstructed = parameterization.positions(zero_parameters)

    np.testing.assert_allclose(
        _periodic_difference(reconstructed, initial),
        0.0,
        atol=1e-10,
    )

    tags = topology.node_types[topology.node_indices]
    rng = np.random.default_rng(19)
    displaced_parameters = rng.normal(
        scale=0.03,
        size=parameterization.n_node_parameters,
    )
    for node_parameters in (zero_parameters, displaced_parameters):
        positions = parameterization.positions(node_parameters)
        for operation, mapping in zip(
            parameterization.operations,
            parameterization.operation_mappings,
        ):
            assert len(np.unique(mapping)) == len(mapping)
            np.testing.assert_array_equal(tags[mapping], tags)
            transformed = np.array(
                [operation.operate(position) for position in positions]
            )
            np.testing.assert_allclose(
                _periodic_difference(transformed, positions[mapping]),
                0.0,
                atol=1e-10,
            )


def test_site_stabilizers_determine_special_and_general_dimensions() -> None:
    parameterization = SymmetryReducedParameterization(
        _make_p4mm_node_topology()
    )
    dimensions = {orbit.tag: orbit.dimension for orbit in parameterization.orbits}

    assert dimensions == {0: 1, 1: 3}
    assert parameterization.n_parameters == (
        sum(dimensions.values()) + parameterization.n_cell_parameters
    )
    assert parameterization.n_parameters == 6

    rng = np.random.default_rng(4)
    for orbit in parameterization.orbits:
        displacement = orbit.basis @ rng.normal(size=orbit.dimension)
        for operation_index in orbit.site_symmetry_operation_indices:
            rotation = parameterization.rotations[operation_index]
            np.testing.assert_allclose(
                rotation @ displacement,
                displacement,
                atol=1e-11,
            )


def test_node_bounds_preserve_fixed_periodic_images() -> None:
    parameterization = SymmetryReducedParameterization(
        _make_p4mm_node_topology()
    )

    for orbit in parameterization.orbits:
        for corner_index in range(1 << orbit.dimension):
            corner = np.array(
                [
                    orbit.bounds[index][(corner_index >> index) & 1]
                    for index in range(orbit.dimension)
                ]
            )
            displacements = np.einsum(
                "mrd,d->mr",
                orbit.jacobians,
                corner,
            )
            assert np.all(np.abs(displacements) < 0.5)

    general = next(orbit for orbit in parameterization.orbits if orbit.tag == 1)
    np.testing.assert_allclose(
        general.bounds,
        np.tile([-0.5 + 1e-8, 0.5 - 1e-8], (3, 1)),
        atol=1e-12,
    )


def test_invariant_cell_strains_preserve_the_declared_metric() -> None:
    parameterization = SymmetryReducedParameterization(
        _make_p4mm_node_topology()
    )
    assert parameterization.n_cell_parameters == 2

    rng = np.random.default_rng(7)
    for _ in range(5):
        cell, _ = parameterization.cell.cell_and_derivatives(
            rng.normal(scale=0.15, size=parameterization.n_cell_parameters)
        )
        metric = cell @ cell.T
        for rotation in parameterization.rotations:
            np.testing.assert_allclose(
                rotation.T @ metric @ rotation,
                metric,
                rtol=1e-11,
                atol=1e-11,
            )


def test_cell_derivatives_match_scipy_frechet_reference() -> None:
    parameterization = SymmetryReducedParameterization(
        _make_p4mm_node_topology()
    )
    cell_parameterization = parameterization.cell
    parameters = np.array([0.17, -0.09])
    strain = np.einsum(
        "k,kij->ij",
        parameters,
        cell_parameterization.basis,
    )

    cell, derivatives = cell_parameterization.cell_and_derivatives(parameters)
    expected_cell = cell_parameterization.initial_cell @ scipy.linalg.expm(
        strain
    )
    expected_derivatives = np.array(
        [
            cell_parameterization.initial_cell
            @ scipy.linalg.expm_frechet(
                strain,
                direction,
                compute_expm=False,
            )
            for direction in cell_parameterization.basis
        ]
    )

    np.testing.assert_allclose(cell, expected_cell, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        derivatives,
        expected_derivatives,
        rtol=1e-12,
        atol=1e-12,
    )


def test_complete_reduced_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(13)
    parameterization = SymmetryReducedParameterization(
        _make_p4mm_node_topology()
    )
    n_nodes = len(parameterization.node_indices)
    n_terms = 16
    ij = rng.integers(0, n_nodes, size=(n_terms, 2))
    ik = rng.integers(0, n_nodes, size=(n_terms, 2))
    ij_image = rng.integers(-1, 2, size=(n_terms, 3))
    ik_image = rng.integers(-1, 2, size=(n_terms, 3))
    target_dots = rng.normal(size=n_terms)
    weights = rng.uniform(0.5, 2.0, size=n_terms)
    x = rng.normal(scale=0.03, size=parameterization.n_parameters)

    def evaluate(values):
        node_values, cell_values = parameterization.split(values)
        positions = parameterization.positions(node_values)
        cell, derivatives = parameterization.cell.cell_and_derivatives(
            cell_values
        )
        value, position_gradient, cell_gradient = (
            _objective_value_and_gradient(
                positions,
                cell,
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
                parameterization.cell.grad(cell_gradient, derivatives),
            ]
        )
        return value, gradient

    _, analytic = evaluate(x)
    step = 1e-6
    numerical = np.empty_like(x)
    for index in range(len(x)):
        offset = np.zeros_like(x)
        offset[index] = step
        numerical[index] = (
            evaluate(x + offset)[0] - evaluate(x - offset)[0]
        ) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-6)


def test_non_p1_scaler_preserves_operations_and_metric(tmp_path: Path) -> None:
    topology = Topology(
        _write_p4mm_cycle_topology(tmp_path / "p4mm-cycle.cgd")
    )
    block = BuildingBlock(_write_linear_block(tmp_path / "linear.xyz"))
    parameterization = SymmetryReducedParameterization(topology)
    assert topology.n_nodes == 4
    assert parameterization.n_node_parameters == 2
    assert parameterization.n_cell_parameters == 2
    bbs = [
        block if index in topology.node_indices else None
        for index in range(topology.n_slots)
    ]
    permutations = [
        np.arange(2) if index in topology.node_indices else None
        for index in range(topology.n_slots)
    ]

    scaled, result = Scaler().scale(
        topology,
        bbs,
        permutations,
        return_result=True,
    )

    assert result.success
    assert np.isfinite(result.fun)
    positions = scaled.atoms.get_scaled_positions(wrap=True)
    tags = scaled.atoms.get_tags()
    cell = np.asarray(scaled.atoms.cell)
    metric = cell @ cell.T
    maximum_residual = 0.0
    for operation in SpaceGroup(topology.spacegroup).symmetry_ops:
        rotation = np.asarray(operation.rotation_matrix, dtype=float)
        np.testing.assert_allclose(
            rotation.T @ metric @ rotation,
            metric,
            rtol=1e-12,
            atol=1e-12,
        )
        mapped = []
        for index, position in enumerate(positions):
            candidates = np.flatnonzero(tags == tags[index])
            differences = _periodic_difference(
                operation.operate(position), positions[candidates]
            )
            residuals = np.linalg.norm(differences @ cell, axis=1)
            best = int(np.argmin(residuals))
            mapped.append(int(candidates[best]))
            maximum_residual = max(
                maximum_residual, float(residuals[best])
            )
        assert len(set(mapped)) == len(mapped)

    assert maximum_residual <= 1e-5
