#!/usr/bin/env python3
"""Generate the balanced 10k single-linker MOF dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

from mofbb.symmetry_cache import write_symmetry_manifest
from single_linker_mof_generator import GenerationConfig, generate_topology_balanced


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT / "lib")
    parser.add_argument("--output", type=Path, default=ROOT / "10k_mofs")
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "generated/building_block_symmetry_cache",
    )
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--strict-symmetry",
        action="store_true",
        help=(
            "reject P1 fallbacks and retain only assignments preserving the "
            "topology's declared symmetry"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-revalidate", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    library = args.library.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    manifest = write_symmetry_manifest(
        library,
        output,
        args.cache.expanduser().resolve(),
        args.workers,
    )
    generate_topology_balanced(
        GenerationConfig(
            manifest=manifest,
            topology_dir=(library / "topology").resolve(),
            output=output,
            count=args.count,
            seed=args.seed,
            strict_symmetry=args.strict_symmetry,
            resume_revalidate=args.resume_revalidate,
        ),
        workers=args.workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
