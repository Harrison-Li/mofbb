"""Screen CIFs before accepting outputs of an external relaxation pipeline."""

import argparse
import json
from pathlib import Path

from ase.io import read

from mofbb.compose.connectivity import validate_3d_connectivity
from mofbb.compose.linkers import validate_single_linker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cifs", nargs="+", type=Path)
    parser.add_argument("--bond-scale", type=float, default=1.2)
    parser.add_argument(
        "--single-linker", action="store_true",
        help="also require one complete molecular linker type after cutting metal bonds",
    )
    args = parser.parse_args(argv)
    failed = False
    for path in args.cifs:
        result = {"path": str(path.resolve())}
        try:
            atoms = read(path)
            result.update(validate_3d_connectivity(atoms, bond_scale=args.bond_scale))
            if args.single_linker:
                result.update(validate_single_linker(atoms, bond_scale=args.bond_scale))
            result["accepted"] = True
        except (OSError, ValueError, RuntimeError) as error:
            result.update(accepted=False, error=str(error))
            failed = True
        print(json.dumps(result))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
