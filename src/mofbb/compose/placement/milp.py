"""Helpers for binary mixed-integer assignment problems."""

from __future__ import annotations

import numpy as np
import scipy.optimize
from scipy.sparse import coo_matrix


class Constraints:
    """Incrementally assemble sparse bounded linear constraints."""

    def __init__(self) -> None:
        self.rows = []
        self.columns = []
        self.values = []
        self.lower = []
        self.upper = []

    def add(
        self,
        indices,
        coefficients,
        minimum: float,
        maximum: float,
    ) -> None:
        row = len(self.lower)
        self.rows.extend([row] * len(indices))
        self.columns.extend(indices)
        self.values.extend(coefficients)
        self.lower.append(minimum)
        self.upper.append(maximum)

    def scipy(self, count: int) -> scipy.optimize.LinearConstraint:
        matrix = coo_matrix(
            (self.values, (self.rows, self.columns)),
            shape=(len(self.lower), count),
        ).tocsr()
        return scipy.optimize.LinearConstraint(matrix, self.lower, self.upper)


def solve_binary(
    costs: np.ndarray,
    bounds: np.ndarray,
    rules: Constraints,
) -> scipy.optimize.OptimizeResult:
    """Minimize a linear objective over bounded binary variables."""

    return scipy.optimize.milp(
        costs,
        integrality=np.ones(len(costs)),
        bounds=scipy.optimize.Bounds(0.0, bounds),
        constraints=rules.scipy(len(costs)),
    )
