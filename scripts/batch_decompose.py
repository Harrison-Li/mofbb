#!/usr/bin/env python3
"""Decompose all CIF files in a folder through the MOFBB Python API."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

from mofbb.decompose import NodeEdgeDecomposer, SingleNodeDecomposer


DECOMPOSERS = {
    "node-edge": NodeEdgeDecomposer,
    "single-node": SingleNodeDecomposer,
}
COUNT_KEYS = (
    "num_nodes",
    "num_linkers",
    "num_metal_nodes",
    "num_organic_nodes",
    "num_edges",
    "num_connecting_bonds",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def find_cifs(input_dir: Path) -> list[Path]:
    """Return CIF files directly inside *input_dir* in stable order."""

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    cifs = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".cif"
    )
    if not cifs:
        raise ValueError(f"No CIF files found in {input_dir}")
    return cifs


def decompose_one(
    cif: Path,
    output_dir: Path,
    decomposition: str,
    solvent_mode: str,
    x_symbol: str,
    bond_scale: float,
    cleanup: bool,
) -> dict[str, object]:
    """Decompose one CIF atomically, returning a compact status record."""

    started = time.monotonic()
    staging_dir: Path | None = None
    record: dict[str, object] = {
        "input": str(cif),
        "output": str(output_dir),
    }
    try:
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Output already exists: {output_dir}")

        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
        )
        decomposer = DECOMPOSERS[decomposition](
            cif,
            x_symbol=x_symbol,
            bond_scale=bond_scale,
            cleanup=cleanup,
            solvent_mode=solvent_mode,
        )
        summary = decomposer.decompose().write(staging_dir)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Output appeared during processing: {output_dir}")
        os.replace(staging_dir, output_dir)
        staging_dir = None

        record.update(
            status="succeeded",
            counts={key: summary[key] for key in COUNT_KEYS if key in summary},
        )
    except Exception as error:  # Keep one bad MOF from stopping the batch.
        record.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)

    record["seconds"] = round(time.monotonic() - started, 3)
    return record


def batch_decompose(
    input_dir: Path,
    output_dir: Path,
    *,
    decomposition: str = "node-edge",
    workers: int = 1,
    solvent_mode: str = "none",
    x_symbol: str = "X",
    bond_scale: float = 1.2,
    cleanup: bool = True,
) -> dict[str, object]:
    """Decompose a directory of CIFs concurrently via the Python API."""

    if decomposition not in DECOMPOSERS:
        raise ValueError(f"Unsupported decomposition: {decomposition}")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    cifs = find_cifs(Path(input_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(cif, output_dir / cif.stem) for cif in cifs]
    results: list[dict[str, object]] = []

    # Completed directories are skipped, making interrupted batches resumable.
    pending = []
    for cif, destination in jobs:
        if (destination / "decomposition_summary.json").is_file():
            results.append(
                {
                    "input": str(cif),
                    "output": str(destination),
                    "status": "skipped",
                    "reason": "completed output already exists",
                    "seconds": 0.0,
                }
            )
        else:
            pending.append((cif, destination))

    def report_progress(record: dict[str, object], completed: int) -> None:
        detail = record.get("error", record["output"])
        print(
            f"[{completed}/{len(jobs)}] {str(record['status']).upper()}: {detail}",
            flush=True,
        )

    for completed, record in enumerate(results, start=1):
        report_progress(record, completed)

    worker_args = (
        decomposition,
        solvent_mode,
        x_symbol,
        bond_scale,
        cleanup,
    )
    if workers == 1:
        for cif, destination in pending:
            record = decompose_one(cif, destination, *worker_args)
            results.append(record)
            report_progress(record, len(results))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    decompose_one, cif, destination, *worker_args
                ): (cif, destination)
                for cif, destination in pending
            }
            for future in as_completed(futures):
                cif, destination = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = {
                        "input": str(cif),
                        "output": str(destination),
                        "status": "failed",
                        "error": f"Worker process failed: {error}",
                        "seconds": 0.0,
                    }
                results.append(record)
                report_progress(record, len(results))

    results.sort(key=lambda item: str(item["input"]))
    report: dict[str, object] = {
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "decomposition": decomposition,
        "workers": workers,
        "total": len(results),
        "succeeded": sum(item["status"] == "succeeded" for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "skipped": sum(item["status"] == "skipped" for item in results),
        "results": results,
    }
    report_path = output_dir / "batch_summary.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--decomposition",
        choices=tuple(DECOMPOSERS),
        default="node-edge",
    )
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument(
        "--solvent-mode", choices=("none", "free", "all"), default="none"
    )
    parser.add_argument("--x-symbol", default="X")
    parser.add_argument("--bond-scale", type=float, default=1.2)
    parser.add_argument("--no-cleanup", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = batch_decompose(
            args.input_dir,
            args.output_dir,
            decomposition=args.decomposition,
            workers=args.workers,
            solvent_mode=args.solvent_mode,
            x_symbol=args.x_symbol,
            bond_scale=args.bond_scale,
            cleanup=not args.no_cleanup,
        )
    except (NotADirectoryError, ValueError) as error:
        parser.error(str(error))
    print(
        f"Finished: {report['succeeded']} succeeded, "
        f"{report['failed']} failed, {report['skipped']} skipped."
    )
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
