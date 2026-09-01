"""Compatibility imports for node-position and lattice optimization."""

from mofbb.compose.node_lattice import (
    NodeLatticeOptimizer,
    _objective_value_and_gradient,
    sp,
)


class Scaler(NodeLatticeOptimizer):
    """Backward-compatible name for :class:`NodeLatticeOptimizer`."""

    def scale(self, topology, bbs, perms, return_result=False):
        """Compatibility wrapper for :meth:`optimize`."""
        return self.optimize(topology, bbs, perms, return_result=return_result)


__all__ = ["NodeLatticeOptimizer", "Scaler", "_objective_value_and_gradient"]
