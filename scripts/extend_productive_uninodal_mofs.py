#!/usr/bin/env python3
"""Extend an existing uninodal dataset from unused strict-symmetry queues.

This is deliberately separate from the balanced generator.  It preserves the
balanced generator's cursors in ``topology_state`` and records extension
cursors in a mode-specific table.  Strict mode selects topologies that reached
quota before relaxation; relaxed mode selects all topologies that reached the
original quota.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, wait
from concurrent.futures.process import BrokenProcessPool
import json
import os
from pathlib import Path
import signal
import shutil
import sqlite3
import time
import warnings

import generate_500k_single_linker_mofs as bulk
import single_linker_mof_generator as core


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_SYMMETRY_MODE = "strict"
MODE = "strict_productive_extension_v1"
PHASE = "strict_productive_extension"
STATE_TABLE = "productive_extension_state"
META_PREFIX = "productive_extension_"
PROGRESS_FILE = "productive_extension_progress.json"
SUMMARY_FILE = "productive_extension_summary.json"
DEFAULT_TARGET = 1_000_000
DEFAULT_WORKERS = 20
STOP_REQUESTED = False


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=positive, default=DEFAULT_TARGET)
    parser.add_argument("--workers", type=positive, default=DEFAULT_WORKERS)
    parser.add_argument("--progress-interval", type=positive, default=30)
    parser.add_argument(
        "--symmetry-mode",
        choices=("strict", "relaxed"),
        default="strict",
        help=(
            "strict requires the detected source space group; relaxed allows "
            "independent/asymmetric placement"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "uninodal_mofs_v1",
        help="existing resumable uninodal dataset",
    )
    return parser.parse_args()


def configure_mode(symmetry_mode: str) -> None:
    global ACTIVE_SYMMETRY_MODE, MODE, PHASE, STATE_TABLE, META_PREFIX
    global PROGRESS_FILE, SUMMARY_FILE
    ACTIVE_SYMMETRY_MODE = symmetry_mode
    if symmetry_mode == "strict":
        MODE = "strict_productive_extension_v1"
        PHASE = "strict_productive_extension"
        STATE_TABLE = "productive_extension_state"
        META_PREFIX = "productive_extension_"
        PROGRESS_FILE = "productive_extension_progress.json"
        SUMMARY_FILE = "productive_extension_summary.json"
    elif symmetry_mode == "relaxed":
        MODE = "relaxed_productive_extension_v1"
        PHASE = "relaxed_productive_extension"
        STATE_TABLE = "productive_relaxed_extension_state"
        META_PREFIX = "productive_relaxed_extension_"
        PROGRESS_FILE = "productive_relaxed_extension_progress.json"
        SUMMARY_FILE = "productive_relaxed_extension_summary.json"
    else:
        raise ValueError(f"unknown productive extension symmetry mode: {symmetry_mode}")


def create_extension_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {STATE_TABLE}(
            name TEXT PRIMARY KEY,
            space_path TEXT NOT NULL,
            start_cursor INTEGER NOT NULL,
            cursor INTEGER NOT NULL,
            size INTEGER NOT NULL,
            baseline_accepted INTEGER NOT NULL,
            accepted_extension INTEGER NOT NULL DEFAULT 0,
            raw_scanned INTEGER NOT NULL DEFAULT 0,
            build_attempts INTEGER NOT NULL DEFAULT 0,
            symmetry_preserving INTEGER NOT NULL DEFAULT 0,
            jobs INTEGER NOT NULL DEFAULT 0,
            exhausted INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.commit()


def extension_meta(connection: sqlite3.Connection, key: str, default=None):
    return bulk.meta(connection, f"{META_PREFIX}{key}", default)


def set_extension_meta(
    connection: sqlite3.Connection, key: str, value: object
) -> None:
    bulk.set_meta(connection, f"{META_PREFIX}{key}", value)


def productive_rows(
    connection: sqlite3.Connection,
) -> list[sqlite3.Row]:
    quota = int(bulk.meta(connection, "per_topology_count", "0") or 0)
    if quota < 1:
        raise RuntimeError("the source dataset has no stored per-topology quota")
    if ACTIVE_SYMMETRY_MODE == "strict":
        rows = connection.execute(
            "SELECT * FROM topology_state WHERE accepted_total>=? "
            "AND phase_attempts=0 AND strict_attempts>0 ORDER BY name",
            (quota,),
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM topology_state WHERE accepted_total>=? "
            "AND inserted_cursor<inserted_size ORDER BY name",
            (quota,),
        ).fetchall()
    if not rows:
        raise RuntimeError(
            f"no {ACTIVE_SYMMETRY_MODE}-productive topology queues were found"
        )
    return rows


def initialize_extension(
    connection: sqlite3.Connection,
    output: Path,
    target: int,
) -> tuple[int, list[str]]:
    create_extension_schema(connection)
    completed = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    maximum_id = int(
        connection.execute("SELECT COALESCE(MAX(id),0) FROM samples").fetchone()[0]
    )
    if completed != maximum_id:
        raise RuntimeError(
            "sample IDs are not contiguous; extension cannot safely allocate IDs"
        )
    if target < completed:
        raise RuntimeError(
            f"target {target} is below the {completed} committed samples"
        )
    if bulk.meta(connection, "topology_mode", "binodal") != "uninodal":
        raise RuntimeError("productive extension requires an uninodal dataset")
    if bulk.meta(connection, "connectivity_policy", "current") != "legacy":
        raise RuntimeError(
            "this extension is configured to match the source dataset's legacy "
            "connectivity policy"
        )

    stored_mode = extension_meta(connection, "mode")
    if stored_mode is None:
        rows = productive_rows(connection)
        state_root = output / ".state" / ACTIVE_SYMMETRY_MODE
        prepared = []
        for row in rows:
            path = state_root / f"{bulk.safe_name(row['name'])}.pickle"
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing {ACTIVE_SYMMETRY_MODE} candidate space: {path}"
                )
            space = bulk.load_space(str(path))
            size = int(space["inserted_size"])
            cursor = int(
                row["raw_scanned"]
                if ACTIVE_SYMMETRY_MODE == "strict"
                else row["inserted_cursor"]
            )
            baseline_accepted = int(
                row["accepted_total"]
                if ACTIVE_SYMMETRY_MODE == "strict"
                else row["phase_successes"]
            )
            if cursor < 0 or cursor > size:
                raise RuntimeError(
                    f"invalid reconstructed {ACTIVE_SYMMETRY_MODE} cursor for "
                    f"{row['name']}: "
                    f"{cursor}/{size}"
                )
            prepared.append((row, path, cursor, size, baseline_accepted))

        with connection:
            for row, path, cursor, size, baseline_accepted in prepared:
                connection.execute(
                    f"INSERT INTO {STATE_TABLE}("
                    "name,space_path,start_cursor,cursor,size,baseline_accepted,"
                    "exhausted) VALUES(?,?,?,?,?,?,?)",
                    (
                        row["name"],
                        str(path),
                        cursor,
                        cursor,
                        size,
                        baseline_accepted,
                        int(cursor >= size),
                    ),
                )
            original_target = bulk.meta(connection, "target_count", str(completed))
            set_extension_meta(connection, "mode", MODE)
            set_extension_meta(connection, "phase", PHASE)
            set_extension_meta(connection, "target", target)
            set_extension_meta(connection, "initial_count", completed)
            set_extension_meta(connection, "active_seconds", 0.0)
            set_extension_meta(connection, "status", "initialized")
            set_extension_meta(
                connection,
                "selected_topologies",
                json.dumps([str(row["name"]) for row in rows]),
            )
            if bulk.meta(connection, "pre_extension_target_count") is None:
                bulk.set_meta(
                    connection, "pre_extension_target_count", original_target
                )
            bulk.set_meta(connection, "target_count", target)
    else:
        if stored_mode != MODE:
            raise RuntimeError(f"unknown stored productive extension mode: {stored_mode}")
        stored_target = int(extension_meta(connection, "target", "0") or 0)
        if target != stored_target:
            raise RuntimeError(
                f"extension target cannot change on resume: {stored_target} != {target}"
            )

    names = [
        str(row[0])
        for row in connection.execute(
            f"SELECT name FROM {STATE_TABLE} ORDER BY name"
        )
    ]
    if not names:
        raise RuntimeError("productive extension state is empty")
    return completed, names


def extension_states(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, object]]:
    return {
        str(row["name"]): dict(row)
        for row in connection.execute(f"SELECT * FROM {STATE_TABLE}")
    }


def observed_raw_yield(state: dict[str, object]) -> float:
    accepted = int(state["baseline_accepted"]) + int(
        state["accepted_extension"]
    )
    scanned = int(state["cursor"])
    return accepted / scanned if scanned else 0.0


def schedule_key(state: dict[str, object]) -> tuple[object, ...]:
    """Prefer queues with demonstrated strict yield, then shorter jobs."""

    return (
        -observed_raw_yield(state),
        int(state["jobs"]),
        state["name"],
    )


def persist_extension(
    connection: sqlite3.Connection,
    output: Path,
    result: dict[str, object],
    completed: int,
    target: int,
) -> int:
    name = str(result["topology"])
    state = connection.execute(
        f"SELECT * FROM {STATE_TABLE} WHERE name=?", (name,)
    ).fetchone()
    if state is None or int(state["cursor"]) != int(result["start"]):
        raise RuntimeError(f"stale productive-extension result for {name}")

    accepted = 0
    symmetry_preserving = 0
    postbuild_mismatches = 0
    connection.execute("BEGIN IMMEDIATE")
    try:
        for success in result["successes"]:
            cif = Path(success["cif"])
            candidate, quality = success["candidate"], success["quality"]
            matches_source = bool(
                quality["detected_matches_source_space_group"]
            )
            if ACTIVE_SYMMETRY_MODE == "strict" and not matches_source:
                postbuild_mismatches += 1
                cif.unlink(missing_ok=True)
                continue
            duplicate = connection.execute(
                "SELECT 1 FROM samples WHERE signature=? OR cif_sha256=? LIMIT 1",
                (candidate.signature, quality["cif_sha256"]),
            ).fetchone()
            if duplicate or completed + accepted >= target:
                cif.unlink(missing_ok=True)
                continue

            sample_number = completed + accepted + 1
            sample_id = f"single_linker_mof_{sample_number:07d}_{name}"
            relative = (
                Path("mofs")
                / f"{(sample_number - 1) // bulk.SHARD_SIZE:05d}"
                / f"{sample_id}.cif"
            )
            final_cif = output / relative
            final_cif.parent.mkdir(parents=True, exist_ok=True)
            os.replace(cif, final_cif)
            nodes = [
                {"node_type": int(node_type), **bulk.block_reference(block)}
                for node_type, block in candidate.nodes
            ]
            edges = [
                {
                    "edge_type": [int(value) for value in edge_type],
                    **bulk.block_reference(block),
                }
                for edge_type, block in candidate.edges
            ]
            quality = dict(quality)
            quality["extension_mode"] = MODE
            quality["extension_phase"] = PHASE
            connection.execute(
                "INSERT INTO samples("
                "id,sample_id,signature,cif_sha256,topology,cif_path,cif_size,"
                "nodes_json,edges_json,quality_json,detected_space_group_number,"
                "generation_tier,generation_phase,created_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sample_number,
                    sample_id,
                    candidate.signature,
                    quality["cif_sha256"],
                    name,
                    str(relative),
                    final_cif.stat().st_size,
                    bulk.dumps(nodes),
                    bulk.dumps(edges),
                    bulk.dumps(quality),
                    int(quality["detected_space_group_number"]),
                    quality["generation_tier"],
                    PHASE,
                    time.time(),
                ),
            )
            accepted += 1
            symmetry_preserving += int(matches_source)

        cursor = int(result["cursor"])
        exhausted = int(cursor >= int(state["size"]))
        connection.execute(
            f"UPDATE {STATE_TABLE} SET cursor=?,"
            "accepted_extension=accepted_extension+?,"
            "raw_scanned=raw_scanned+?,build_attempts=build_attempts+?,"
            "symmetry_preserving=symmetry_preserving+?,jobs=jobs+1,"
            "exhausted=? WHERE name=?",
            (
                cursor,
                accepted,
                int(result["raw"]),
                int(result["attempted"]),
                symmetry_preserving,
                exhausted,
                name,
            ),
        )
        if ACTIVE_SYMMETRY_MODE == "strict":
            connection.execute(
                "UPDATE topology_state SET accepted_total=accepted_total+?,"
                "raw_scanned=raw_scanned+?,strict_attempts=strict_attempts+?,"
                "strict_exact=strict_exact+? WHERE name=?",
                (
                    accepted,
                    int(result["raw"]),
                    int(result["strict_attempts"]),
                    symmetry_preserving,
                    name,
                ),
            )
        else:
            connection.execute(
                "UPDATE topology_state SET accepted_total=accepted_total+?,"
                "inserted_cursor=?,raw_scanned=raw_scanned+?,"
                "phase_raw=phase_raw+?,phase_attempts=phase_attempts+?,"
                "phase_successes=phase_successes+?,"
                "relaxed_attempts=relaxed_attempts+?,exhausted=? WHERE name=?",
                (
                    accepted,
                    cursor,
                    int(result["raw"]),
                    int(result["raw"]),
                    int(result["attempted"]),
                    accepted,
                    int(result["relaxed_attempts"]),
                    exhausted,
                    name,
                ),
            )
        failures = dict(result["failures"])
        if postbuild_mismatches:
            failures["PostbuildSpaceGroupMismatch"] = (
                failures.get("PostbuildSpaceGroupMismatch", 0)
                + postbuild_mismatches
            )
        for error_type, count in failures.items():
            connection.execute(
                "INSERT INTO failure_counts(topology,phase,error_type,count) "
                "VALUES(?,?,?,?) ON CONFLICT(topology,phase,error_type) "
                "DO UPDATE SET count=count+excluded.count",
                (name, PHASE, str(error_type), int(count)),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return accepted


def progress_payload(
    connection: sqlite3.Connection,
    target: int,
    previous_seconds: float,
    started: float,
    status: str,
) -> dict[str, object]:
    completed = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    initial = int(extension_meta(connection, "initial_count", str(completed)))
    extension_count = int(
        connection.execute(
            "SELECT COALESCE(SUM(accepted_extension),0) "
            f"FROM {STATE_TABLE}"
        ).fetchone()[0]
    )
    state = connection.execute(
        "SELECT COALESCE(SUM(raw_scanned),0),"
        "COALESCE(SUM(build_attempts),0),"
        "COALESCE(SUM(symmetry_preserving),0),"
        "COALESCE(SUM(exhausted),0),"
        f"COALESCE(SUM(size-cursor),0) FROM {STATE_TABLE}"
    ).fetchone()
    p1 = int(
        connection.execute(
            "SELECT COUNT(*) FROM samples WHERE detected_space_group_number=1"
        ).fetchone()[0]
    )
    elapsed = previous_seconds + time.monotonic() - started
    extension_rate = extension_count / elapsed if elapsed else 0.0
    symmetry_policy = (
        "strict_source_space_group_preserving"
        if ACTIVE_SYMMETRY_MODE == "strict"
        else "relaxed_asymmetric_allowed"
    )
    return {
        "status": status,
        "generation_phase": PHASE,
        "symmetry_policy": symmetry_policy,
        "target": target,
        "completed": completed,
        "remaining": max(0, target - completed),
        "initial_count": initial,
        "extension_completed": extension_count,
        "extension_raw_positions_scanned": int(state[0]),
        "extension_build_attempts": int(state[1]),
        "extension_symmetry_preserving": int(state[2]),
        "extension_exhausted_topologies": int(state[3]),
        "extension_candidate_positions_remaining": int(state[4]),
        "selected_productive_topologies": int(
            connection.execute(
                f"SELECT COUNT(*) FROM {STATE_TABLE}"
            ).fetchone()[0]
        ),
        "accepted_p1_count": p1,
        "current_p1_fraction": p1 / completed if completed else 0.0,
        "elapsed_extension_seconds": elapsed,
        "extension_accepted_per_hour": extension_rate * 3600,
        "estimated_remaining_extension_seconds": (
            (target - completed) / extension_rate
            if extension_rate and completed < target
            else 0.0
        ),
        "updated_at": time.time(),
    }


def save_progress(
    connection: sqlite3.Connection,
    output: Path,
    target: int,
    previous_seconds: float,
    started: float,
    status: str,
) -> dict[str, object]:
    payload = progress_payload(
        connection, target, previous_seconds, started, status
    )
    bulk.write_json(output / PROGRESS_FILE, payload)
    bulk.write_json(output / "progress.json", payload)
    return payload


def finish(
    connection: sqlite3.Connection,
    output: Path,
    target: int,
    previous_seconds: float,
    started: float,
    status: str,
) -> dict[str, object]:
    elapsed = previous_seconds + time.monotonic() - started
    with connection:
        set_extension_meta(connection, "active_seconds", elapsed)
        set_extension_meta(connection, "status", status)
    payload = save_progress(
        connection, output, target, previous_seconds, started, status
    )
    bulk.write_json(output / SUMMARY_FILE, payload)
    return payload


def run_extension(
    connection: sqlite3.Connection,
    config: core.GenerationConfig,
    output: Path,
    target: int,
    workers_requested: int,
    progress_interval: int,
) -> tuple[int, str]:
    completed = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    if completed >= target:
        return completed, "complete"
    previous_seconds = float(extension_meta(connection, "active_seconds", "0") or 0)
    started = time.monotonic()
    last_progress = 0.0
    workers = workers_requested
    pool_failures = 0
    jobs_done = 0
    futures: dict[object, dict[str, object]] = {}
    inflight: set[str] = set()
    states = extension_states(connection)
    executor = bulk.executor_for(config, workers)
    status = "running"

    def recover_pool(error: BrokenProcessPool, abandoned: int) -> None:
        nonlocal executor, pool_failures, workers, states, jobs_done
        if jobs_done >= bulk.HEALTHY_POOL_JOBS:
            pool_failures = 0
        pool_failures += 1
        futures.clear()
        inflight.clear()
        bulk.terminate(executor)
        if pool_failures > bulk.MAX_POOL_FAILURES:
            raise RuntimeError("extension worker pool repeatedly terminated") from error
        workers = max(1, workers_requested - 2 * (pool_failures - 1))
        executor = bulk.executor_for(config, workers)
        states = extension_states(connection)
        jobs_done = 0
        print(
            f"extension_worker_pool_recovered={pool_failures} "
            f"abandoned_jobs={abandoned} workers={workers}",
            flush=True,
        )

    try:
        while completed < target and not STOP_REQUESTED:
            while len(futures) < workers and jobs_done < bulk.RESTART_JOBS:
                choices = [
                    state
                    for state in states.values()
                    if state["name"] not in inflight
                    and not int(state["exhausted"])
                    and int(state["cursor"]) < int(state["size"])
                ]
                if not choices:
                    break
                state = min(choices, key=schedule_key)
                batch = min(bulk.ATTEMPT_BATCH, target - completed)
                job = {
                    "topology": state["name"],
                    "space_path": state["space_path"],
                    "mode": "inserted",
                    "cursor": state["cursor"],
                    "batch": batch,
                    "strict": ACTIVE_SYMMETRY_MODE == "strict",
                    "staging": str(output / ".staging" / PHASE),
                }
                try:
                    future = executor.submit(bulk.build_chunk, job)
                except BrokenProcessPool as error:
                    recover_pool(error, len(futures))
                    continue
                futures[future] = job
                inflight.add(str(state["name"]))

            if jobs_done >= bulk.RESTART_JOBS and not futures:
                executor.shutdown(wait=True, cancel_futures=True)
                workers = workers_requested
                executor = bulk.executor_for(config, workers)
                jobs_done = pool_failures = 0
                print(f"extension_worker_pool_recycled workers={workers}", flush=True)
                continue
            if not futures:
                status = "exhausted"
                break

            if STOP_REQUESTED:
                status = "interrupted"
                bulk.terminate(executor)
                futures.clear()
                inflight.clear()
                break
            done, _ = wait(
                tuple(futures), timeout=1.0, return_when=FIRST_COMPLETED
            )
            if not done:
                continue
            broken = False
            for future in done:
                job = futures.pop(future)
                name = str(job["topology"])
                inflight.discard(name)
                try:
                    result = future.result()
                except BrokenProcessPool as error:
                    recover_pool(error, len(futures) + 1)
                    broken = True
                    break
                accepted = persist_extension(
                    connection, output, result, completed, target
                )
                completed += accepted
                jobs_done += 1
                states[name] = dict(
                    connection.execute(
                        f"SELECT * FROM {STATE_TABLE} WHERE name=?",
                        (name,),
                    ).fetchone()
                )
                now = time.monotonic()
                if now - last_progress >= progress_interval or completed >= target:
                    payload = save_progress(
                        connection,
                        output,
                        target,
                        previous_seconds,
                        started,
                        "complete" if completed >= target else "running",
                    )
                    print(
                        f"extension_progress={completed}/{target} phase={PHASE} "
                        f"topology={name} accepted={accepted}/{result['attempted']} "
                        f"extension_rate_per_hour="
                        f"{payload['extension_accepted_per_hour']:.1f}",
                        flush=True,
                    )
                    last_progress = now
            if broken:
                continue

        if STOP_REQUESTED:
            status = "interrupted"
            if futures:
                bulk.terminate(executor)
        elif completed >= target:
            status = "complete"
            bulk.terminate(executor)
        else:
            executor.shutdown(wait=True, cancel_futures=True)
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        bulk.terminate(executor)
        finish(
            connection,
            output,
            target,
            previous_seconds,
            started,
            status,
        )
        raise

    finish(
        connection,
        output,
        target,
        previous_seconds,
        started,
        status,
    )
    return completed, status


def interrupt_handler(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main() -> None:
    # The relaxed builder commonly encounters underdetermined vector fits.  They
    # are valid for this mode and otherwise produce many megabytes of warnings.
    warnings.filterwarnings(
        "ignore",
        message=r"Optimal rotation is not uniquely or poorly defined.*",
        category=UserWarning,
    )
    args = arguments()
    configure_mode(args.symmetry_mode)
    output = args.output.expanduser().resolve()
    database = output / "generation.sqlite3"
    manifest = output / "block_symmetry_manifest.csv"
    if not database.is_file() or not manifest.is_file():
        raise FileNotFoundError(
            f"existing generation database and manifest are required in {output}"
        )
    signal.signal(signal.SIGINT, interrupt_handler)
    signal.signal(signal.SIGTERM, interrupt_handler)

    connection = bulk.connect(database)
    completed, names = initialize_extension(
        connection, output, args.target
    )
    config = core.GenerationConfig(
        manifest=manifest,
        topology_dir=(ROOT / "lib" / "topology").resolve(),
        output=output,
        count=args.target,
        seed=bulk.DEFAULT_SEED,
        edge_conformer_filter="all",
        strict_symmetry=ACTIVE_SYMMETRY_MODE == "strict",
        strict_prefilter=True,
        require_3d_connectivity=False,
        attempt_batch_size=bulk.ATTEMPT_BATCH,
        worker_memory_limit_gib=bulk.WORKER_MEMORY_GIB,
    )
    states = extension_states(connection)
    remaining = sum(int(row["size"]) - int(row["cursor"]) for row in states.values())
    require_source = ACTIVE_SYMMETRY_MODE == "strict"
    print(
        f"productive_extension_mode={MODE} symmetry={ACTIVE_SYMMETRY_MODE} "
        f"require_source_space_group={str(require_source).lower()} "
        f"selected_topologies={len(names)} "
        f"completed={completed}/{args.target} remaining_candidates={remaining}",
        flush=True,
    )
    completed, status = run_extension(
        connection,
        config,
        output,
        args.target,
        args.workers,
        args.progress_interval,
    )
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    if status == "complete":
        shutil.rmtree(output / ".staging" / PHASE, ignore_errors=True)
    print(
        f"productive_extension_status={status} "
        f"completed={completed}/{args.target} output={output}",
        flush=True,
    )
    if completed != args.target:
        raise RuntimeError(
            f"productive {ACTIVE_SYMMETRY_MODE} extension generated "
            f"{completed} of {args.target}"
        )


if __name__ == "__main__":
    main()
