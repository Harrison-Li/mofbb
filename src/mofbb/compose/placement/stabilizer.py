"""Building-block compatibility with crystallographic site stabilizers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.spatial.transform import Rotation


class StabilizerFit:
    """Measure site-symmetry residuals and optimize linker torsion."""

    def __init__(self, symmetry):
        self.symmetry = symmetry

    def diagnose(self, orbit, block, perm, cell, tol):
        """Measure motif compatibility with an orbit stabilizer."""
        if tol <= 0:
            raise ValueError("Stabilizer tolerance must be positive.")

        pos = block.atoms.positions - block.centroid
        symbols = np.asarray(block.atoms.get_chemical_symbols())
        ports = np.asarray(block.connection_point_indices, int)
        perm = np.asarray(perm, int)
        atom_err = port_err = 0.0

        for raw_op in orbit.stabilizers:
            op = int(raw_op)
            moved = pos @ self.symmetry.transform(op, cell)
            atom_err = max(atom_err, self.species_error(moved, pos, symbols))
            offsets = self.symmetry.offset_map(
                orbit.representative,
                orbit.representative,
                op,
            )
            if perm.shape != offsets.shape:
                raise ValueError(
                    f"Slot {orbit.representative} has {len(offsets)} neighbors, "
                    f"but permutation shape is {perm.shape}."
                )
            src = ports[perm]
            dst = ports[perm[offsets]]
            errors = np.linalg.norm(moved[src] - pos[dst], axis=1)
            port_err = max(port_err, float(np.max(errors, initial=0)))

        error = max(atom_err, port_err)
        return {
            "tag": int(orbit.tag),
            "representative": int(orbit.representative),
            "orbit_size": int(len(orbit.members)),
            "stabilizer_order": int(len(orbit.stabilizers)),
            "tolerance": float(tol),
            "max_atom_residual": float(atom_err),
            "max_port_residual": float(port_err),
            "compatible": error <= tol,
        }

    def fit(self, orbit, block, perm, cell, tol):
        """Fit the free linker-axis torsion to an edge-slot stabilizer."""
        if not block.is_edge:
            raise ValueError("Stabilizer torsion optimization requires an edge.")

        base = self.diagnose(orbit, block, perm, cell, tol)
        error = max(base["max_atom_residual"], base["max_port_residual"])
        base.update(
            torsion_optimized=False,
            torsion_angle_degrees=0.0,
            best_torsion_residual=float(error),
        )
        if base["compatible"] or len(orbit.stabilizers) == 1:
            return block, base

        perm = np.asarray(perm, int)
        ports = block.connection_points[perm]
        axis = ports[1] - ports[0]
        axis /= np.linalg.norm(axis)
        center = block.centroid
        pos = block.atoms.positions - center
        trial = block.copy()

        def evaluate(angle):
            angle = float(angle % self.symmetry.tau)
            moved = Rotation.from_rotvec(angle * axis).apply(pos) + center
            trial.atoms.set_positions(moved)
            report = self.diagnose(orbit, trial, perm, cell, tol)
            error = max(
                report["max_atom_residual"],
                report["max_port_residual"],
            )
            return float(error), moved, report, angle

        step = self.symmetry.tau / self.symmetry.torsion_steps
        samples = [
            evaluate(i * step) for i in range(self.symmetry.torsion_steps)
        ]
        low = min(item[0] for item in samples)
        best = min(
            (item for item in samples if item[0] <= low + 1.0e-10),
            key=lambda item: item[3],
        )
        if best[0] > tol:
            result = minimize_scalar(
                lambda angle: evaluate(angle)[0],
                bounds=(best[3] - step, best[3] + step),
                method="bounded",
                options={"xatol": 1.0e-5, "maxiter": 24},
            )
            best = min(best, evaluate(result.x), key=lambda item: item[0])

        base["best_torsion_residual"] = best[0]
        if best[0] > tol:
            return block, base

        fitted = block.copy()
        fitted.atoms.set_positions(best[1])
        report = best[2]
        report.update(
            torsion_optimized=True,
            torsion_angle_degrees=float(np.degrees(best[3])),
            best_torsion_residual=best[0],
        )
        return fitted, report

    @staticmethod
    def species_error(moved, ref, symbols):
        """Return the largest optimal same-species assignment distance."""
        error = 0.0
        for symbol in np.unique(symbols):
            ids = np.flatnonzero(symbols == symbol)
            dists = np.linalg.norm(
                moved[ids, None] - ref[None, ids],
                axis=2,
            )
            rows, cols = linear_sum_assignment(dists)
            error = max(error, float(np.max(dists[rows, cols], initial=0)))
        return error
