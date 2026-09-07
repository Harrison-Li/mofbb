"""Screen CIFs before accepting outputs of an external relaxation pipeline."""

import argparse
import json
from pathlib import Path

from ase.io import read

from mofbb.compose.connectivity import validate_3d_connectivity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cifs", nargs="+", type=Path)
    parser.add_argument("--bond-scale", type=float, default=1.2)
    args = parser.parse_args(argv)
    failed = False
    for path in args.cifs:
        result = {"path": str(path.resolve())}
        try:
            result.update(validate_3d_connectivity(read(path), bond_scale=args.bond_scale))
            result["accepted"] = True
        except (OSError, ValueError, RuntimeError) as error:
            result.update(accepted=False, error=str(error))
            failed = True
        print(json.dumps(result))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
