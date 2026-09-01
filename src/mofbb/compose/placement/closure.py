"""Compatibility imports for the renamed fixed-cell fitting stage."""

from mofbb.compose.placement.fixed_cell import FixedCellFrameworkFitter


class CellClosure(FixedCellFrameworkFitter):
    """Backward-compatible name for :class:`FixedCellFrameworkFitter`."""

    @property
    def closed(self):
        return self.fitted

    @closed.setter
    def closed(self, value):
        self.fitted = value

    @property
    def fit(self):
        return self.optimization_result

    @fit.setter
    def fit(self, value):
        self.optimization_result = value

    @property
    def covariant_node_orbits(self):
        return self.symmetry_compatible_node_orbits

    @covariant_node_orbits.setter
    def covariant_node_orbits(self, value):
        self.symmetry_compatible_node_orbits = value

    @property
    def covariant_edge_orbits(self):
        return self.symmetry_compatible_edge_orbits

    @covariant_edge_orbits.setter
    def covariant_edge_orbits(self, value):
        self.symmetry_compatible_edge_orbits = value


__all__ = ["CellClosure", "FixedCellFrameworkFitter"]
