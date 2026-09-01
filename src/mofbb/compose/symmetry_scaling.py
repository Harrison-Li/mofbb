"""Compatibility imports for geometry parameterization."""

from mofbb.compose.geometry_parameterization import (
    SiteOrbit,
    Symmetry,
    SymmetryConstrainedNodeLatticeParameterization,
    SymmetryInvariantLatticeParameterization,
)

# Backward-compatible aliases for callers using the pre-0.2 names.
InvariantCellParameterization = SymmetryInvariantLatticeParameterization
SymmetryReducedParameterization = SymmetryConstrainedNodeLatticeParameterization

__all__ = [
    "InvariantCellParameterization",
    "SiteOrbit",
    "Symmetry",
    "SymmetryConstrainedNodeLatticeParameterization",
    "SymmetryInvariantLatticeParameterization",
    "SymmetryReducedParameterization",
]
