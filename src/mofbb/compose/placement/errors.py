"""Exceptions raised by symmetry-aware placement."""

from __future__ import annotations


class SymmetryCompatibilityError(ValueError):
    """Reject a build that cannot preserve its declared slot symmetry."""

    def __init__(
        self,
        reason,
        *,
        kind,
        representative,
        member=None,
        diagnostic=None,
    ):
        self.reason = str(reason)
        self.kind = str(kind)
        self.representative = int(representative)
        self.member = None if member is None else int(member)
        self.diagnostic = diagnostic

        location = (
            f"{self.kind} orbit representative {self.representative}"
        )
        if self.member is not None:
            location += f", member {self.member}"
        details = []
        if diagnostic is not None:
            for key in (
                "max_atom_residual",
                "max_port_residual",
                "best_torsion_residual",
                "propagated_port_residual",
            ):
                value = diagnostic.get(key)
                if value is not None:
                    details.append(f"{key}={float(value):.6g}")
        suffix = f" ({', '.join(details)})" if details else ""
        super().__init__(
            "Strict symmetry rejected "
            f"{location}: {self.reason}{suffix}."
        )
