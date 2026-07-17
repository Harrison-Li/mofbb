#!/usr/bin/env python3
"""Benchmark topology-driven MOF reconstruction in mofbb and PORMAKE.

Both backends read the same PORMAKE CGD and XYZ files. Input parsing is kept
outside the timed region. Each backend/case pair runs in a fresh subprocess so
imports and runtime caches cannot leak from one implementation into the other.

mofbb's geometric gates are still evaluated, but their rejection thresholds
are relaxed because PORMAKE does not apply equivalent rejection gates.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


RESULT_PREFIX = "MOF_RECONSTRUCTION_BENCHMARK="
BACKENDS = ("mofbb", "pormake")
THREAD_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "JAX_PLATFORM_NAME": "cpu",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    topology: str
    node_blocks: tuple[tuple[int, str], ...]
    edge_blocks: tuple[tuple[tuple[int, int], str], ...] = ()


CASES = (
    BenchmarkCase("pcu_N16_E19", "pcu", ((0, "N16"),), (((0, 0), "E19"),)),
    BenchmarkCase("pcu_N16_E27", "pcu", ((0, "N16"),), (((0, 0), "E27"),)),
    BenchmarkCase("pcu_N16_E38", "pcu", ((0, "N16"),), (((0, 0), "E38"),)),
    BenchmarkCase("pcu_N16_E41", "pcu", ((0, "N16"),), (((0, 0), "E41"),)),
    BenchmarkCase("pcu_N295_E19", "pcu", ((0, "N295"),), (((0, 0), "E19"),)),
    BenchmarkCase("HKUST-1", "tbo", ((0, "N10"), (1, "N409"))),
    BenchmarkCase(
        "tbo_N10_N409_E41",
        "tbo",
        ((0, "N10"), (1, "N409")),
        (((0, 1), "E41"),),
    ),
    BenchmarkCase(
        "ith_N3_N114_E41",
        "ith",
        ((0, "N3"), (1, "N114")),
        (((0, 1), "E41"),),
    ),
)
CASES_BY_NAME = {case.name: case for case in CASES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
        help="Number of measured builds per backend and case (default: 7).",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=2,
        help="Number of unmeasured builds per backend and case (default: 2).",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(CASES_BY_NAME),
        dest="case_names",
        help="Benchmark only this case; may be supplied more than once.",
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        help="PORMAKE database directory containing topologies/ and bbs/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the complete JSON result.",
    )
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-case",
        choices=tuple(CASES_BY_NAME),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def pormake_database_root() -> Path:
    spec = importlib.util.find_spec("pormake")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("PORMAKE is not importable.")
    return Path(next(iter(spec.submodule_search_locations))) / "database"


def configure_quiet_logging() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Optimal rotation is not uniquely or poorly defined.*",
    )
    for module_name in ("mofbb.compose.log", "pormake.log"):
        try:
            module = __import__(module_name, fromlist=["logger"])
        except ImportError:
            continue
        logger = module.logger
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL)


def load_backend_inputs(backend: str, case: BenchmarkCase, database_root: Path):
    if backend == "mofbb":
        from mofbb.compose import Builder, BuildingBlock, Topology

        build_kwargs = {"scale": 10.0, "tolerance": 10.0}
    else:
        from pormake import Builder, BuildingBlock, Topology

        build_kwargs = {}

    topology = Topology(database_root / "topologies" / f"{case.topology}.cgd")
    nodes = {
        node_type: BuildingBlock(database_root / "bbs" / f"{name}.xyz")
        for node_type, name in case.node_blocks
    }
    edges = {
        edge_type: BuildingBlock(database_root / "bbs" / f"{name}.xyz")
        for edge_type, name in case.edge_blocks
    }

    def build():
        return Builder().build_by_type(topology, nodes, edges, **build_kwargs)

    return topology, build


def timed_build(build):
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    start = time.perf_counter_ns()
    try:
        framework = build()
    finally:
        elapsed_seconds = (time.perf_counter_ns() - start) / 1.0e9
        if gc_was_enabled:
            gc.enable()
    return elapsed_seconds, framework


def framework_summary(framework) -> dict[str, object]:
    return {
        "atoms": len(framework.atoms),
        "bonds": len(framework.bonds),
        "cell_parameters": [
            float(value) for value in framework.atoms.cell.cellpar()
        ],
        "formula": framework.atoms.get_chemical_formula(),
    }


def run_worker(
    backend: str,
    case: BenchmarkCase,
    database_root: Path,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    configure_quiet_logging()
    topology, build = load_backend_inputs(backend, case, database_root)

    cold_seconds = None
    output = None
    for warmup_index in range(warmups):
        elapsed_seconds, output = timed_build(build)
        if warmup_index == 0:
            cold_seconds = elapsed_seconds

    samples = []
    for _ in range(repeats):
        elapsed_seconds, output = timed_build(build)
        samples.append(elapsed_seconds)

    assert output is not None
    return {
        "backend": backend,
        "case": case.name,
        "cold_seconds": cold_seconds,
        "mean_seconds": statistics.fmean(samples),
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "stdev_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples_seconds": samples,
        "framework": framework_summary(output),
        "topology": {
            "name": case.topology,
            "spacegroup": str(topology.spacegroup),
            "nodes": int(topology.n_nodes),
            "edges": int(topology.n_edges),
            "slots": int(topology.n_slots),
        },
    }


def run_subprocess(
    backend: str,
    case: BenchmarkCase,
    database_root: Path,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        backend,
        "--worker-case",
        case.name,
        "--database-root",
        str(database_root),
        "--warmups",
        str(warmups),
        "--repeats",
        str(repeats),
    ]
    environment = os.environ.copy()
    environment.update(THREAD_ENVIRONMENT)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{backend}/{case.name} failed with exit code "
            f"{completed.returncode}:\n{completed.stderr}{completed.stdout}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    raise RuntimeError(
        f"{backend}/{case.name} did not emit a benchmark result:\n"
        f"{completed.stderr}{completed.stdout}"
    )


def validate_matching_outputs(results: list[dict[str, object]]) -> None:
    for case_name in {result["case"] for result in results}:
        outputs = [
            result["framework"] for result in results if result["case"] == case_name
        ]
        comparable = [
            (output["atoms"], output["bonds"], output["formula"])
            for output in outputs
        ]
        if len(set(comparable)) != 1:
            raise RuntimeError(f"Backend outputs differ for {case_name}: {outputs}")


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled-worktree"


def git_state() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def format_relative_speed(ratio: float) -> str:
    if ratio >= 1.0:
        return f"{ratio:.2f}x faster"
    return f"{1.0 / ratio:.2f}x slower"


def print_summary(results: list[dict[str, object]]) -> None:
    indexed = {(result["case"], result["backend"]): result for result in results}
    print("\nWarm build timings (median seconds)")
    print("case                    slots  atoms   PORMAKE     mofbb   mofbb relative")
    print("----------------------  -----  -----  --------  --------  --------------")
    pormake_total = 0.0
    mofbb_total = 0.0
    for case in CASES:
        key = (case.name, "pormake")
        if key not in indexed:
            continue
        pormake_result = indexed[key]
        mofbb_result = indexed[(case.name, "mofbb")]
        pormake_seconds = pormake_result["median_seconds"]
        mofbb_seconds = mofbb_result["median_seconds"]
        pormake_total += pormake_seconds
        mofbb_total += mofbb_seconds
        speed_ratio = pormake_seconds / mofbb_seconds
        print(
            f"{case.name:22s}  "
            f"{pormake_result['topology']['slots']:5d}  "
            f"{pormake_result['framework']['atoms']:5d}  "
            f"{pormake_seconds:8.4f}  "
            f"{mofbb_seconds:8.4f}  "
            f"{format_relative_speed(speed_ratio):>14s}"
        )
    total_ratio = pormake_total / mofbb_total
    print(
        f"{'suite total':22s}  {'':5s}  {'':5s}  "
        f"{pormake_total:8.4f}  {mofbb_total:8.4f}  "
        f"{format_relative_speed(total_ratio):>14s}"
    )


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.warmups < 1:
        raise SystemExit("--warmups must be at least 1 so cold timing is defined")

    database_root = args.database_root or pormake_database_root()
    if not (database_root / "topologies").is_dir() or not (
        database_root / "bbs"
    ).is_dir():
        raise SystemExit(f"Invalid PORMAKE database root: {database_root}")

    if args.worker:
        if args.worker_case is None:
            raise SystemExit("--worker-case is required with --worker")
        result = run_worker(
            args.worker,
            CASES_BY_NAME[args.worker_case],
            database_root,
            args.warmups,
            args.repeats,
        )
        print(RESULT_PREFIX + json.dumps(result, separators=(",", ":")))
        return

    selected_cases = (
        [CASES_BY_NAME[name] for name in args.case_names]
        if args.case_names
        else list(CASES)
    )
    results = []
    for case in selected_cases:
        for backend in BACKENDS:
            print(f"Running {backend:8s} {case.name} ...", flush=True)
            results.append(
                run_subprocess(
                    backend,
                    case,
                    database_root,
                    args.warmups,
                    args.repeats,
                )
            )

    validate_matching_outputs(results)
    document = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": cpu_model(),
            "logical_cpu_affinity": len(os.sched_getaffinity(0)),
            "numeric_threads": 1,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "timed_region": "Builder().build_by_type with preloaded inputs",
            "mofbb_quality_gate_thresholds": {
                "scale": 10.0,
                "tolerance_angstrom": 10.0,
            },
            "versions": {
                "mofbb": package_version("mofbbr"),
                "pormake": package_version("pormake"),
            },
            "mofbb_git": git_state(),
        },
        "results": results,
    }
    print_summary(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
