'''
Exceptions raised by symmetry-aware placement.
'''

from __future__ import annotations


class SymmetryCompatibilityError(ValueError):
    '''
    Report a placement that cannot preserve the declared slot symmetry.

    Args:
        reason (str):
            Short identifier for the failed symmetry check.
        kind (str):
            Type of failed slot or check, such as node, edge, or topology.
        representative (int):
            Representative slot of the failed orbit.
        member (int or None):
            Specific member slot, when the failure is limited to one member.
        diagnostic (dict or None):
            Residuals and other details included in the error message when available.
    '''

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
