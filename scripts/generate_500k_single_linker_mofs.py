#!/usr/bin/env python3
"""Generate a large single-linker MOF dataset, strict-first and resumably."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import signal
import sqlite3
import time
import warnings

from threadpoolctl import threadpool_limits

from mofbb.symmetry_cache import write_symmetry_manifest
import single_linker_mof_generator as core


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 2
SPACE_VERSION = 2
DEFAULT_COUNT = 2_000_000
DEFAULT_WORKERS = min(64, os.cpu_count() or 1)
DEFAULT_SEED = 42
SHARD_SIZE = 10_000
ATTEMPT_BATCH = 250
SCAN_BATCH = 10_000
SELECTION_POOL = 8
PILOT_ATTEMPTS = 1_000
PILOT_RAW = 100_000
MINIMUM_YIELD = 0.01
RESTART_JOBS = 256
HEALTHY_POOL_JOBS = 64
WORKER_MEMORY_GIB = 16
MAX_POOL_FAILURES = 8
SYMPREC = 0.3

_SPACE_CONFIG: core.GenerationConfig | None = None
_SPACE_POOLS: dict[tuple, list] | None = None
_BUILD_CONFIG: core.GenerationConfig | None = None
_STRICT_CONFIG: core.GenerationConfig | None = None


def dumps(value: object, indent: int | None = None) -> str:
    return json.dumps(
        value,
        default=str,
        indent=indent,
        sort_keys=indent is not None,
        separators=None if indent is not None else (",", ":"),
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(dumps(value, 2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_name(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "topology"
    return f"{label}_{hashlib.sha256(value.encode()).hexdigest()[:10]}"


def affine(seed: int, topology: str, label: str, size: int) -> tuple[int, int]:
    if size <= 1:
        return 0, 1
    digest = hashlib.sha256(f"{seed}:{topology}:{label}".encode()).digest()
    offset = int.from_bytes(digest[:16], "big") % size
    step = int.from_bytes(digest[16:], "big") % size or 1
    while math.gcd(step, size) != 1:
        step = step + 1 if step + 1 < size else 1
    return offset, step


def make_space(
    config: core.GenerationConfig,
    topology: core.TopologyRecord,
    pools: dict[tuple, list],
) -> dict[str, object]:
    """Build a compact Cartesian space with cheap strict component screening."""

    strict = config.strict_symmetry
    node_kinds = (
        ("metal_node",)
        if len(topology.node_sites) == 1
        else ("metal_node", "organic_node")
    )
    working = pools
    if strict:
        working, _ = core.strict_node_compatible_pools(
            config,
            topology,
            pools,
            kinds=node_kinds,
        )
        working, _ = core.strict_edge_compatible_pools(topology, working)

    fitted: dict[tuple[int, str], tuple[tuple[core.BlockRecord, float], ...]] = {}
    for site in topology.node_sites:
        for kind in node_kinds:
            values = []
            if strict:
                records = working.get(
                    (kind, site.node_type, site.coordination, site.point_group), ()
                )
                for item in records:
                    values.append((item.block, float(item.rmsd)))
            else:
                # Once symmetry is relaxed, a special crystallographic site
                # may use any coordination-compatible block that passes the
                # same geometric RMSD gate.  This gives low-cardinality
                # uninodal point-group buckets enough distinct assignments to
                # reach a balanced quota without weakening geometry checks.
                records = (
                    working.get(
                        (kind, site.coordination, core.ALL_NODE_POINT_GROUP),
                        (),
                    )
                    if len(topology.node_sites) == 1
                    else core.node_records_for_site(working, kind, site)
                )
                for block in records:
                    fit = core.screened_port_fit_rmsd(
                        site, block, config.max_rmsd, config.accuracy
                    )
                    if fit <= config.max_rmsd:
                        values.append((block, float(fit)))
            fitted[(site.node_type, kind)] = tuple(values)

    edges = []
    for edge_type in topology.edge_types:
        records = working.get(
            ("edge", edge_type, 2, core.EDGE_POINT_GROUP)
            if strict
            else ("edge", 2, core.EDGE_POINT_GROUP),
            (),
        )
        for block in records:
            edges.append((edge_type, block))

    assignment_mode = "binodal"
    uninodal = None
    orientations = []
    if len(topology.node_sites) == 1:
        assignment_mode = "uninodal_self_edge"
        site = topology.node_sites[0]
        metals = fitted[(site.node_type, "metal_node")]
        uninodal = (site, metals)
        direct_sizes = ()
        inserted_sizes = (len(metals) * len(edges),)
    else:
        for metal_site in topology.node_sites:
            for organic_site in topology.node_sites:
                if metal_site.node_type == organic_site.node_type:
                    continue
                metal = fitted[(metal_site.node_type, "metal_node")]
                organic = fitted[(organic_site.node_type, "organic_node")]
                if metal and organic:
                    orientations.append((metal_site, organic_site, metal, organic))
        direct_sizes = tuple(
            len(metal) * len(organic) for _, _, metal, organic in orientations
        )
        inserted_sizes = tuple(size * len(edges) for size in direct_sizes)
    direct_size, inserted_size = sum(direct_sizes), sum(inserted_sizes)
    direct_offset, direct_step = affine(
        config.seed, topology.name, "direct", direct_size
    )
    inserted_offset, inserted_step = affine(
        config.seed, topology.name, "inserted", inserted_size
    )
    return {
        "version": SPACE_VERSION,
        "topology": topology,
        "assignment_mode": assignment_mode,
        "uninodal": uninodal,
        "orientations": tuple(orientations),
        "edges": tuple(edges),
        "direct_sizes": direct_sizes,
        "inserted_sizes": inserted_sizes,
        "direct_size": direct_size,
        "inserted_size": inserted_size,
        "direct_offset": direct_offset,
        "direct_step": direct_step,
        "inserted_offset": inserted_offset,
        "inserted_step": inserted_step,
    }


def init_space_worker(config: core.GenerationConfig, pools: dict[tuple, list]) -> None:
    global _SPACE_CONFIG, _SPACE_POOLS
    _SPACE_CONFIG, _SPACE_POOLS = config, pools
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    core.cached_building_block = lru_cache(maxsize=256)(
        core.cached_building_block.__wrapped__
    )
    core.cached_topology = lru_cache(maxsize=8)(core.cached_topology.__wrapped__)
    core.configure_quiet_builder_logging()


def prepare_space(topology: core.TopologyRecord, path: Path) -> tuple:
    if _SPACE_CONFIG is None or _SPACE_POOLS is None:
        raise RuntimeError("candidate-space worker was not initialized")
    started = time.monotonic()
    with threadpool_limits(limits=1, user_api="blas"):
        space = make_space(_SPACE_CONFIG, topology, _SPACE_POOLS)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pickle.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(space, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return (
        topology.name,
        str(path),
        int(space["direct_size"]),
        int(space["inserted_size"]),
        time.monotonic() - started,
    )


@lru_cache(maxsize=16)
def load_space(path: str) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        space = pickle.load(handle)
    if space.get("version") != SPACE_VERSION:
        raise RuntimeError(f"candidate-space version mismatch: {path}")
    return space


def init_build_worker(config: core.GenerationConfig) -> None:
    global _BUILD_CONFIG, _STRICT_CONFIG
    _BUILD_CONFIG = config
    _STRICT_CONFIG = replace(config, strict_symmetry=True)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # A parent may install a graceful SIGTERM handler before forking.  Workers
    # must retain the default action so ProcessPoolExecutor can terminate an
    # in-flight build promptly without hanging the parent at shutdown.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    core.cached_building_block = lru_cache(maxsize=1_024)(
        core.cached_building_block.__wrapped__
    )
    core.cached_topology = lru_cache(maxsize=16)(core.cached_topology.__wrapped__)
    core.connection_port_compatibility = lru_cache(maxsize=8_192)(
        core.connection_port_compatibility.__wrapped__
    )
    core.configure_generation_worker(config.worker_memory_limit_gib)


@lru_cache(maxsize=32_768)
def exact_node(
    topology: core.TopologyRecord, node_type: int, block: core.BlockRecord
) -> bool:
    try:
        return core.strict_node_record_fit(
            _STRICT_CONFIG, topology, node_type, block
        ) is not None
    except (IndexError, OSError, StopIteration, ValueError):
        return False


@lru_cache(maxsize=32_768)
def exact_edge(
    topology: core.TopologyRecord,
    edge_type: tuple[int, int],
    block: core.BlockRecord,
) -> bool:
    try:
        return core.strict_edge_record_compatible(
            _STRICT_CONFIG, topology, edge_type, block
        )
    except (IndexError, OSError, StopIteration, ValueError):
        return False


def exact_candidate(
    topology: core.TopologyRecord, candidate: core.CandidateAssignment
) -> bool:
    return all(exact_node(topology, key, block) for key, block in candidate.nodes) and all(
        exact_edge(topology, key, block) for key, block in candidate.edges
    )


def locate(space: dict[str, object], mode: str, index: int) -> tuple[tuple, int]:
    for orientation, size in zip(
        space["orientations"], space[f"{mode}_sizes"], strict=True
    ):
        if index < size:
            return orientation, index
        index -= size
    raise IndexError("candidate index exceeds its Cartesian space")


def candidate_at(
    space: dict[str, object], mode: str, ordinal: int
) -> core.CandidateAssignment | None:
    if _BUILD_CONFIG is None:
        raise RuntimeError("generation worker was not initialized")
    total = int(space[f"{mode}_size"])
    if ordinal >= total:
        return None
    index = (
        int(space[f"{mode}_offset"]) + ordinal * int(space[f"{mode}_step"])
    ) % total
    if space.get("assignment_mode") == "uninodal_self_edge":
        if mode != "inserted":
            return None
        site, metals = space["uninodal"]
        edge_options = space["edges"]
        edge_type, edge = edge_options[index % len(edge_options)]
        metal, metal_fit = metals[index // len(edge_options)]
        if (
            not edge.contains_carbon
            or metal.sha256 == edge.sha256
            or metal.structure_id == edge.structure_id
        ):
            return None
        estimated_atoms = (
            int(space["topology"].num_nodes)
            * (int(metal.num_atoms) - int(metal.coordination))
            + int(space["topology"].num_edges)
            * (int(edge.num_atoms) - int(edge.coordination))
        )
        if (
            _BUILD_CONFIG.max_output_atoms
            and estimated_atoms > _BUILD_CONFIG.max_output_atoms
        ):
            return None
        nodes = {site.node_type: metal}
        edges = {edge_type: edge}
        risk = core.assignment_prebuild_risk(
            space["topology"],
            nodes,
            edges,
            maximum_rmsd=_BUILD_CONFIG.max_rmsd,
            accuracy=_BUILD_CONFIG.accuracy,
        )
        if risk is None:
            return None
        return core.CandidateAssignment(
            signature=core.candidate_assignment_signature(
                space["topology"].name, nodes, edges
            ),
            nodes=tuple(sorted(nodes.items())),
            edges=tuple(sorted(edges.items())),
            score=(
                float(risk),
                float(metal_fit),
                max(metal.port_radius_span, edge.port_radius_span),
            ),
        )

    orientation, local = locate(space, mode, index)
    metal_site, organic_site, metals, organics = orientation
    edges: dict[tuple[int, int], core.BlockRecord] = {}
    if mode == "inserted":
        edge_options = space["edges"]
        edge_type, edge = edge_options[local % len(edge_options)]
        edges[edge_type] = edge
        local //= len(edge_options)

    organic, organic_fit = organics[local % len(organics)]
    metal, metal_fit = metals[local // len(organics)]
    if (
        not organic.contains_carbon
        or metal.sha256 == organic.sha256
        or metal.structure_id == organic.structure_id
    ):
        return None
    if edges:
        edge = next(iter(edges.values()))
        if edge.sha256 in {metal.sha256, organic.sha256} or edge.structure_id in {
            metal.structure_id,
            organic.structure_id,
        }:
            return None

    nodes = {metal_site.node_type: metal, organic_site.node_type: organic}
    risk = core.assignment_prebuild_risk(
        space["topology"],
        nodes,
        edges,
        maximum_rmsd=_BUILD_CONFIG.max_rmsd,
        accuracy=_BUILD_CONFIG.accuracy,
    )
    if risk is None:
        return None
    return core.CandidateAssignment(
        signature=core.candidate_assignment_signature(
            space["topology"].name, nodes, edges
        ),
        nodes=tuple(sorted(nodes.items())),
        edges=tuple(sorted(edges.items())),
        score=(
            float(risk),
            max(float(metal_fit), float(organic_fit)),
            max(block.port_radius_span for block in nodes.values()),
        ),
    )


def tier(quality: dict[str, object], strict: bool) -> str:
    if strict:
        if quality["detected_matches_source_space_group"]:
            return "symmetry_preserving"
        return (
            "strict_detected_p1"
            if int(quality["detected_space_group_number"]) == 1
            else "strict_non_source_group"
        )
    non_p1 = int(quality["detected_space_group_number"]) != 1
    if not non_p1:
        return "symmetry_relaxed_p1"
    return (
        "partially_symmetric"
        if int(quality["symmetry_fallback_orbit_count"]) <= 1
        else "symmetry_relaxed_non_p1"
    )


def build_chunk(job: dict[str, object]) -> dict[str, object]:
    if _BUILD_CONFIG is None or _STRICT_CONFIG is None:
        raise RuntimeError("generation worker was not initialized")
    space = load_space(str(job["space_path"]))
    topology = space["topology"]
    mode, strict = str(job["mode"]), bool(job["strict"])
    start, batch = int(job["cursor"]), int(job["batch"])
    stop = min(int(space[f"{mode}_size"]), start + SCAN_BATCH)
    limit = batch if strict else batch * SELECTION_POOL
    candidates, cursor = [], start

    with threadpool_limits(limits=1, user_api="blas"):
        for ordinal in range(start, stop):
            cursor = ordinal + 1
            candidate = candidate_at(space, mode, ordinal)
            if candidate is not None and (not strict or exact_candidate(topology, candidate)):
                candidates.append(candidate)
                if len(candidates) == limit:
                    break

        selected = sorted(candidates, key=lambda item: item.score)[:batch]
        staging = Path(job["staging"]) / safe_name(topology.name)
        staging.mkdir(parents=True, exist_ok=True)
        built_topology = core.cached_topology(str(topology.cgd_path))
        successes, failures = [], Counter()
        for candidate in selected:
            cif = staging / f"{candidate.signature}.cif"
            try:
                quality = core.build_and_validate(
                    _STRICT_CONFIG if strict else _BUILD_CONFIG,
                    topology,
                    {key: block.path for key, block in candidate.nodes},
                    {key: block.path for key, block in candidate.edges},
                    cif,
                    topology=built_topology,
                    crystal_symmetry_symprec=SYMPREC,
                )
                quality["generation_phase"] = "strict" if strict else "relaxed"
                quality["generation_tier"] = tier(quality, strict)
                quality["cif_sha256"] = core.file_sha256(cif)
                successes.append(
                    {"candidate": candidate, "cif": str(cif), "quality": quality}
                )
            except Exception as error:
                cif.unlink(missing_ok=True)
                failures[type(error).__name__] += 1

    return {
        "topology": topology.name,
        "mode": mode,
        "start": start,
        "cursor": cursor,
        "raw": cursor - start,
        "attempted": len(selected),
        "successes": successes,
        "failures": dict(failures),
        "strict_attempts": len(selected) if strict else 0,
        "strict_exact": sum(
            item["quality"]["detected_matches_source_space_group"]
            for item in successes
            if strict
        ),
        "relaxed_attempts": len(selected) if not strict else 0,
    }


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=120.0)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA foreign_keys=ON;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS topology_state(
            name TEXT PRIMARY KEY,
            space_path TEXT NOT NULL,
            direct_cursor INTEGER NOT NULL DEFAULT 0,
            direct_size INTEGER NOT NULL,
            inserted_cursor INTEGER NOT NULL DEFAULT 0,
            inserted_size INTEGER NOT NULL,
            next_mode TEXT NOT NULL DEFAULT 'direct',
            raw_scanned INTEGER NOT NULL DEFAULT 0,
            phase_raw INTEGER NOT NULL DEFAULT 0,
            phase_attempts INTEGER NOT NULL DEFAULT 0,
            phase_successes INTEGER NOT NULL DEFAULT 0,
            accepted_total INTEGER NOT NULL DEFAULT 0,
            strict_attempts INTEGER NOT NULL DEFAULT 0,
            strict_exact INTEGER NOT NULL DEFAULT 0,
            relaxed_attempts INTEGER NOT NULL DEFAULT 0,
            jobs INTEGER NOT NULL DEFAULT 0,
            parked INTEGER NOT NULL DEFAULT 0,
            exhausted INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS samples(
            id INTEGER PRIMARY KEY,
            sample_id TEXT UNIQUE NOT NULL,
            signature TEXT UNIQUE NOT NULL,
            cif_sha256 TEXT UNIQUE NOT NULL,
            topology TEXT NOT NULL,
            cif_path TEXT NOT NULL,
            cif_size INTEGER NOT NULL,
            nodes_json TEXT NOT NULL,
            edges_json TEXT NOT NULL,
            quality_json TEXT NOT NULL,
            detected_space_group_number INTEGER NOT NULL,
            generation_tier TEXT NOT NULL,
            generation_phase TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS samples_topology ON samples(topology);
        CREATE INDEX IF NOT EXISTS samples_space_group
            ON samples(detected_space_group_number);
        CREATE TABLE IF NOT EXISTS failure_counts(
            topology TEXT NOT NULL,
            phase TEXT NOT NULL,
            error_type TEXT NOT NULL,
            count INTEGER NOT NULL,
            PRIMARY KEY(topology, phase, error_type)
        );
        """
    )
    connection.commit()
    return connection


def meta(connection: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return str(row[0]) if row else default


def set_meta(connection: sqlite3.Connection, key: str, value: object) -> None:
    connection.execute(
        "INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def fingerprint(
    config: core.GenerationConfig,
    topologies: list[core.TopologyRecord],
    topology_mode: str = "binodal",
    per_topology_count: int | None = None,
) -> str:
    digest = hashlib.sha256()
    settings = {
        "schema": SCHEMA_VERSION,
        "space": SPACE_VERSION,
        "single_linker_policy": core.SINGLE_LINKER_POLICY_VERSION,
    }
    if config.require_3d_connectivity:
        settings["connectivity_policy"] = core.CONNECTIVITY_POLICY_VERSION
    settings.update({
        "seed": config.seed,
        "max_topology_slots": config.max_topology_slots,
        "edge_filter": config.edge_conformer_filter,
        "node_atoms": config.max_node_atoms,
        "edge_atoms": config.max_edge_atoms,
        "output_atoms": config.max_output_atoms,
        "rmsd": config.max_rmsd,
        "atom_distance": config.minimum_atom_distance,
        "accuracy": config.accuracy,
        "builder_symmetry_tolerance": config.symmetry_tolerance,
        "postbuild_symprec": SYMPREC,
        "strict_component_filter": "lazy_exact",
    })
    if topology_mode != "binodal" or per_topology_count is not None:
        settings.update(
            {
                "topology_mode": topology_mode,
                "per_topology_count": per_topology_count,
            }
        )
    if topology_mode == "uninodal":
        settings.update(
            {
                "uninodal_formulation": "metal_node_required_self_edge_v1",
                "relaxed_uninodal_node_pool": "coordination_plus_rmsd_v1",
            }
        )
    digest.update(
        dumps(settings).encode()
    )
    digest.update(core.file_sha256(config.manifest).encode())
    for topology in topologies:
        digest.update(topology.name.encode())
        digest.update(core.file_sha256(topology.cgd_path).encode())
    return digest.hexdigest()


def initialize(
    connection: sqlite3.Connection,
    source_fingerprint: str,
    count: int,
    relax: bool,
    start_phase: str,
    topology_mode: str = "binodal",
    per_topology_count: int | None = None,
    connectivity_policy: str = "current",
    max_output_atoms: int = 0,
) -> bool:
    version = meta(connection, "schema_version")
    if version is None:
        for key, value in {
            "schema_version": SCHEMA_VERSION,
            "source_fingerprint": source_fingerprint,
            "target_count": count,
            "generation_phase": start_phase,
            "start_phase": start_phase,
            "topology_mode": topology_mode,
            "connectivity_policy": connectivity_policy,
            "max_output_atoms": max_output_atoms,
            "per_topology_count": (
                "" if per_topology_count is None else per_topology_count
            ),
            "candidate_space_phase": "",
            "relax_after_strict_exhaustion": int(relax),
            "reserve_mode": 0,
            "active_seconds": 0.0,
            "created_at": time.time(),
        }.items():
            set_meta(connection, key, value)
        connection.commit()
        return True
    if int(version) != SCHEMA_VERSION:
        raise RuntimeError("output database belongs to an older script; use a fresh folder")
    if meta(connection, "source_fingerprint") != source_fingerprint:
        raise RuntimeError(
            "library, cache, topology, or structural/connectivity/linker settings changed; "
            "use a fresh output folder for the current validation policy"
        )
    if meta(connection, "start_phase", "strict") != start_phase:
        raise RuntimeError("--start-phase cannot change on resume")
    if meta(connection, "topology_mode", "binodal") != topology_mode:
        raise RuntimeError("--topology-mode cannot change on resume")
    stored_connectivity_policy = meta(connection, "connectivity_policy")
    if (
        stored_connectivity_policy is not None
        and stored_connectivity_policy != connectivity_policy
    ):
        raise RuntimeError("--connectivity-policy cannot change on resume")
    if int(meta(connection, "max_output_atoms", "0") or 0) != max_output_atoms:
        raise RuntimeError("--max-output-atoms cannot change on resume")
    stored_quota = meta(connection, "per_topology_count", "")
    requested_quota = "" if per_topology_count is None else str(per_topology_count)
    if stored_quota != requested_quota:
        raise RuntimeError("--per-topology-count cannot change on resume")
    if int(meta(connection, "relax_after_strict_exhaustion", "0")) != int(relax):
        raise RuntimeError("--relax-after-strict-exhaustion cannot change on resume")
    completed = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    if count < completed:
        raise RuntimeError(f"requested count {count} is below {completed} saved MOFs")
    if connection.execute(
        "SELECT COUNT(*) FROM samples WHERE detected_space_group_number IS NULL"
    ).fetchone()[0]:
        raise RuntimeError("saved samples without symmetry classification were found")
    set_meta(connection, "target_count", count)
    set_meta(connection, "connectivity_policy", connectivity_policy)
    connection.commit()
    return False


def prepare_spaces(
    connection: sqlite3.Connection,
    config: core.GenerationConfig,
    pools: dict[tuple, list],
    topologies: list[core.TopologyRecord],
    directory: Path,
    workers: int,
) -> None:
    existing = {
        row["name"]: row for row in connection.execute("SELECT * FROM topology_state")
    }
    jobs = []
    for topology in topologies:
        path = directory / f"{safe_name(topology.name)}.pickle"
        row = existing.get(topology.name)
        if row is None or Path(row["space_path"]) != path or not path.is_file():
            jobs.append((topology, path))

    if jobs:
        process_count = min(workers, 32, len(jobs))
        print(f"candidate_space_plan={len(jobs)} workers={process_count}", flush=True)
        with ProcessPoolExecutor(
            max_workers=process_count,
            initializer=init_space_worker,
            initargs=(config, pools),
        ) as executor:
            futures = [
                executor.submit(prepare_space, topology, path)
                for topology, path in jobs
            ]
            for done, future in enumerate(as_completed(futures), 1):
                name, path, direct, inserted, seconds = future.result()
                if max(direct, inserted) >= 2**63:
                    raise OverflowError(f"candidate space exceeds SQLite INTEGER: {name}")
                connection.execute(
                    "INSERT INTO topology_state(name,space_path,direct_size,inserted_size) "
                    "VALUES(?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
                    "space_path=excluded.space_path,direct_size=excluded.direct_size,"
                    "inserted_size=excluded.inserted_size",
                    (name, path, direct, inserted),
                )
                connection.commit()
                print(
                    f"candidate_space={done}/{len(jobs)} topology={name} "
                    f"direct={direct} inserted={inserted} seconds={seconds:.2f}",
                    flush=True,
                )

    expected = {topology.name for topology in topologies}
    stored = {row[0] for row in connection.execute("SELECT name FROM topology_state")}
    if stored != expected:
        raise RuntimeError("stored and eligible topology sets differ")


def prepare_phase(
    connection: sqlite3.Connection,
    config: core.GenerationConfig,
    pools: dict[tuple, list],
    topologies: list[core.TopologyRecord],
    state_root: Path,
    workers: int,
    phase: str,
) -> None:
    prepare_spaces(
        connection,
        replace(config, strict_symmetry=phase == "strict"),
        pools,
        topologies,
        state_root / phase,
        workers,
    )
    if meta(connection, "candidate_space_phase") == phase:
        return
    connection.execute(
        "UPDATE topology_state SET direct_cursor=0,inserted_cursor=0,"
        "next_mode='direct',phase_raw=0,phase_attempts=0,phase_successes=0,"
        "jobs=0,parked=0,exhausted=CASE WHEN direct_size=0 AND inserted_size=0 "
        "THEN 1 ELSE 0 END"
    )
    set_meta(connection, "candidate_space_phase", phase)
    set_meta(connection, "reserve_mode", 0)
    connection.commit()


def block_reference(block: core.BlockRecord) -> dict[str, object]:
    return {
        "block_sha256": block.sha256,
        "kind": block.kind,
        "structure_id": block.structure_id,
        "source_path": str(block.path),
    }


def persist(
    connection: sqlite3.Connection,
    output: Path,
    result: dict[str, object],
    completed: int,
    target: int,
    phase: str,
    per_topology_count: int | None = None,
) -> int:
    name, mode = str(result["topology"]), str(result["mode"])
    state = connection.execute(
        "SELECT * FROM topology_state WHERE name=?", (name,)
    ).fetchone()
    if state is None or int(state[f"{mode}_cursor"]) != int(result["start"]):
        raise RuntimeError(f"stale result for {name}/{mode}")

    accepted = 0
    connection.execute("BEGIN IMMEDIATE")
    try:
        for success in result["successes"]:
            cif = Path(success["cif"])
            candidate, quality = success["candidate"], success["quality"]
            duplicate = connection.execute(
                "SELECT 1 FROM samples WHERE signature=? OR cif_sha256=? LIMIT 1",
                (candidate.signature, quality["cif_sha256"]),
            ).fetchone()
            topology_quota_reached = (
                per_topology_count is not None
                and int(state["accepted_total"]) + accepted >= per_topology_count
            )
            if (
                duplicate
                or completed + accepted >= target
                or topology_quota_reached
            ):
                cif.unlink(missing_ok=True)
                continue

            sample_number = completed + accepted + 1
            sample_id = f"single_linker_mof_{sample_number:07d}_{name}"
            relative = (
                Path("mofs")
                / f"{(sample_number - 1) // SHARD_SIZE:05d}"
                / f"{sample_id}.cif"
            )
            final_cif = output / relative
            final_cif.parent.mkdir(parents=True, exist_ok=True)
            os.replace(cif, final_cif)
            nodes = [
                {"node_type": int(node_type), **block_reference(block)}
                for node_type, block in candidate.nodes
            ]
            edges = [
                {
                    "edge_type": [int(value) for value in edge_type],
                    **block_reference(block),
                }
                for edge_type, block in candidate.edges
            ]
            connection.execute(
                "INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sample_number,
                    sample_id,
                    candidate.signature,
                    quality["cif_sha256"],
                    name,
                    str(relative),
                    final_cif.stat().st_size,
                    dumps(nodes),
                    dumps(edges),
                    dumps(quality),
                    int(quality["detected_space_group_number"]),
                    quality["generation_tier"],
                    phase,
                    time.time(),
                ),
            )
            accepted += 1

        direct_cursor = (
            int(result["cursor"]) if mode == "direct" else int(state["direct_cursor"])
        )
        inserted_cursor = (
            int(result["cursor"])
            if mode == "inserted"
            else int(state["inserted_cursor"])
        )
        exhausted = int(
            direct_cursor >= int(state["direct_size"])
            and inserted_cursor >= int(state["inserted_size"])
        )
        connection.execute(
            f"UPDATE topology_state SET {mode}_cursor=?,next_mode=?,"
            "raw_scanned=raw_scanned+?,phase_raw=phase_raw+?,"
            "phase_attempts=phase_attempts+?,phase_successes=phase_successes+?,"
            "accepted_total=accepted_total+?,strict_attempts=strict_attempts+?,"
            "strict_exact=strict_exact+?,relaxed_attempts=relaxed_attempts+?,"
            "jobs=jobs+1,exhausted=? WHERE name=?",
            (
                result["cursor"],
                "inserted" if mode == "direct" else "direct",
                result["raw"],
                result["raw"],
                result["attempted"],
                accepted,
                accepted,
                result["strict_attempts"],
                result["strict_exact"],
                result["relaxed_attempts"],
                exhausted,
                name,
            ),
        )
        for error_type, count in result["failures"].items():
            connection.execute(
                "INSERT INTO failure_counts VALUES(?,?,?,?) "
                "ON CONFLICT(topology,phase,error_type) DO UPDATE SET "
                "count=count+excluded.count",
                (name, phase, error_type, int(count)),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return accepted


def states(connection: sqlite3.Connection) -> dict[str, dict[str, object]]:
    return {
        row["name"]: dict(row)
        for row in connection.execute("SELECT * FROM topology_state")
    }


def available_mode(state: dict[str, object]) -> str | None:
    preferred = str(state["next_mode"])
    other = "inserted" if preferred == "direct" else "direct"
    for mode in (preferred, other):
        if int(state[f"{mode}_cursor"]) < int(state[f"{mode}_size"]):
            return mode
    return None


def below_topology_quota(
    state: dict[str, object], per_topology_count: int | None
) -> bool:
    return (
        per_topology_count is None
        or int(state["accepted_total"]) < per_topology_count
    )


def low_yield(state: dict[str, object]) -> bool:
    attempts = int(state["phase_attempts"])
    if attempts >= PILOT_ATTEMPTS:
        return core.wilson_upper_bound(
            int(state["phase_successes"]), attempts
        ) < MINIMUM_YIELD
    return int(state["phase_raw"]) >= PILOT_RAW and attempts < 10


def schedule_key(state: dict[str, object]) -> tuple[object, ...]:
    attempts, raw = int(state["phase_attempts"]), int(state["phase_raw"])
    if attempts < PILOT_ATTEMPTS and raw < PILOT_RAW:
        return (0, int(state["jobs"]), attempts, state["name"])
    observed = int(state["phase_successes"]) / max(1, attempts)
    return (1, int(state["accepted_total"]), -observed, int(state["jobs"]), state["name"])


def progress(
    connection: sqlite3.Connection,
    target: int,
    previous_seconds: float,
    started: float,
    status: str,
) -> dict[str, object]:
    completed = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    raw, attempts, parked, exhausted = connection.execute(
        "SELECT COALESCE(SUM(raw_scanned),0),"
        "COALESCE(SUM(strict_attempts+relaxed_attempts),0),"
        "COALESCE(SUM(parked),0),COALESCE(SUM(exhausted),0) "
        "FROM topology_state"
    ).fetchone()
    failures = int(
        connection.execute("SELECT COALESCE(SUM(count),0) FROM failure_counts").fetchone()[0]
    )
    p1 = int(
        connection.execute(
            "SELECT COUNT(*) FROM samples WHERE detected_space_group_number=1"
        ).fetchone()[0]
    )
    elapsed = previous_seconds + time.monotonic() - started
    rate = completed / elapsed if elapsed else 0.0
    return {
        "status": status,
        "generation_phase": meta(connection, "generation_phase"),
        "target": target,
        "completed": completed,
        "remaining": max(0, target - completed),
        "raw_cartesian_positions_scanned": int(raw),
        "build_attempts": int(attempts),
        "failed_build_attempts": failures,
        "parked_topologies": int(parked),
        "exhausted_topologies": int(exhausted),
        "accepted_p1_count": p1,
        "current_p1_fraction": p1 / completed if completed else 0.0,
        "elapsed_active_seconds": elapsed,
        "accepted_per_active_hour": rate * 3600,
        "estimated_remaining_active_seconds": (
            (target - completed) / rate if rate and completed < target else 0.0
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
    payload = progress(connection, target, previous_seconds, started, status)
    write_json(output / "progress.json", payload)
    return payload


def terminate(executor: ProcessPoolExecutor) -> None:
    for process in list((getattr(executor, "_processes", None) or {}).values()):
        if process.is_alive():
            process.terminate()
    executor.shutdown(wait=False, cancel_futures=True)


def executor_for(config: core.GenerationConfig, workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=workers, initializer=init_build_worker, initargs=(config,)
    )


def finish_timing(
    connection: sqlite3.Connection,
    output: Path,
    target: int,
    previous_seconds: float,
    started: float,
    status: str,
) -> None:
    set_meta(
        connection,
        "active_seconds",
        previous_seconds + time.monotonic() - started,
    )
    connection.commit()
    save_progress(connection, output, target, previous_seconds, started, status)


def run_phase(
    connection: sqlite3.Connection,
    config: core.GenerationConfig,
    output: Path,
    args: argparse.Namespace,
    phase: str,
) -> tuple[int, str]:
    completed = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    if completed >= args.count:
        return completed, "complete"
    topology_states = states(connection)
    previous_seconds = float(meta(connection, "active_seconds", "0") or 0)
    started, last_progress = time.monotonic(), 0.0
    reserve = meta(connection, "reserve_mode", "0") == "1"
    workers, pool_failures, jobs_done = args.workers, 0, 0
    futures, inflight = {}, set()
    executor = executor_for(config, workers)
    status = "running"
    per_topology_count = getattr(args, "per_topology_count", None)

    def recover_pool(error: BrokenProcessPool, abandoned: int) -> None:
        nonlocal executor, pool_failures, workers, topology_states, jobs_done
        if jobs_done >= HEALTHY_POOL_JOBS:
            pool_failures = 0
        pool_failures += 1
        futures.clear()
        inflight.clear()
        terminate(executor)
        if pool_failures > MAX_POOL_FAILURES:
            raise RuntimeError("worker pool repeatedly terminated") from error
        workers = max(1, args.workers - 2 * (pool_failures - 1))
        executor = executor_for(config, workers)
        topology_states, jobs_done = states(connection), 0
        print(
            f"worker_pool_recovered={pool_failures} "
            f"abandoned_jobs={abandoned} workers={workers}",
            flush=True,
        )

    try:
        while completed < args.count:
            while len(futures) < workers and jobs_done < RESTART_JOBS:
                choices = [
                    state
                    for state in topology_states.values()
                    if state["name"] not in inflight
                    and not int(state["parked"])
                    and not int(state["exhausted"])
                    and below_topology_quota(state, per_topology_count)
                    and available_mode(state) is not None
                ]
                if not choices:
                    break
                state = min(choices, key=schedule_key)
                mode = available_mode(state)
                batch = min(
                    ATTEMPT_BATCH,
                    max(25, math.ceil((args.count - completed) / workers) * 2),
                )
                if per_topology_count is not None:
                    batch = min(
                        batch,
                        per_topology_count - int(state["accepted_total"]),
                    )
                job = {
                    "topology": state["name"],
                    "space_path": state["space_path"],
                    "mode": mode,
                    "cursor": state[f"{mode}_cursor"],
                    "batch": batch,
                    "strict": phase == "strict",
                    "staging": str(output / ".staging"),
                }
                try:
                    future = executor.submit(build_chunk, job)
                except BrokenProcessPool as error:
                    recover_pool(error, len(futures))
                    continue
                futures[future] = job
                inflight.add(state["name"])

            if jobs_done >= RESTART_JOBS and not futures:
                executor.shutdown(wait=True, cancel_futures=True)
                workers = args.workers
                executor = executor_for(config, workers)
                jobs_done = pool_failures = 0
                print(f"worker_pool_recycled workers={workers}", flush=True)
                continue
            if not futures:
                parked = [
                    state
                    for state in topology_states.values()
                    if int(state["parked"])
                    and not int(state["exhausted"])
                    and below_topology_quota(state, per_topology_count)
                    and available_mode(state) is not None
                ]
                if parked and not reserve:
                    connection.execute("UPDATE topology_state SET parked=0 WHERE exhausted=0")
                    set_meta(connection, "reserve_mode", 1)
                    connection.commit()
                    topology_states, reserve = states(connection), True
                    print("low_yield_reserve_mode=enabled", flush=True)
                    continue
                status = "strict_exhausted" if phase == "strict" else "exhausted"
                break

            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            broken = False
            for future in done:
                job = futures.pop(future)
                name = str(job["topology"])
                inflight.discard(name)
                try:
                    result = future.result()
                except BrokenProcessPool as error:
                    abandoned = len(futures) + 1
                    recover_pool(error, abandoned)
                    broken = True
                    break

                accepted = persist(
                    connection,
                    output,
                    result,
                    completed,
                    args.count,
                    phase,
                    per_topology_count,
                )
                completed += accepted
                jobs_done += 1
                topology_states[name] = dict(
                    connection.execute(
                        "SELECT * FROM topology_state WHERE name=?", (name,)
                    ).fetchone()
                )
                if not reserve and low_yield(topology_states[name]):
                    connection.execute(
                        "UPDATE topology_state SET parked=1 WHERE name=?", (name,)
                    )
                    connection.commit()
                    topology_states[name]["parked"] = 1

                now = time.monotonic()
                if now - last_progress >= args.progress_interval or completed >= args.count:
                    payload = save_progress(
                        connection,
                        output,
                        args.count,
                        previous_seconds,
                        started,
                        "complete" if completed >= args.count else "running",
                    )
                    print(
                        f"progress={completed}/{args.count} phase={phase} topology={name} "
                        f"accepted={accepted}/{result['attempted']} "
                        f"p1_fraction={payload['current_p1_fraction']:.4f} "
                        f"rate_per_hour={payload['accepted_per_active_hour']:.1f}",
                        flush=True,
                    )
                    last_progress = now
            if broken:
                continue

        if completed >= args.count:
            status = "complete"
            terminate(executor)
        else:
            executor.shutdown(wait=True, cancel_futures=True)
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        terminate(executor)
        finish_timing(
            connection, output, args.count, previous_seconds, started, status
        )
        raise

    finish_timing(connection, output, args.count, previous_seconds, started, status)
    return completed, status


def finalize(
    connection: sqlite3.Connection,
    output: Path,
    config: core.GenerationConfig,
    args: argparse.Namespace,
    status: str,
) -> dict[str, object]:
    rows = [dict(row) for row in connection.execute("SELECT * FROM topology_state")]
    completed = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    attempts = sum(
        int(row["strict_attempts"]) + int(row["relaxed_attempts"]) for row in rows
    )
    p1 = int(
        connection.execute(
            "SELECT COUNT(*) FROM samples WHERE detected_space_group_number=1"
        ).fetchone()[0]
    )
    failure_counts = {
        f"{row['phase']}:{row['error_type']}": int(row["count"])
        for row in connection.execute(
            "SELECT phase,error_type,SUM(count) AS count FROM failure_counts "
            "GROUP BY phase,error_type ORDER BY phase,error_type"
        )
    }
    topology_counts = {
        row["name"]: int(row["accepted_total"])
        for row in sorted(rows, key=lambda item: item["name"])
    }
    per_topology_count = getattr(args, "per_topology_count", None)
    topology_shortfalls = (
        {
            name: per_topology_count - count
            for name, count in topology_counts.items()
            if count < per_topology_count
        }
        if per_topology_count is not None
        else {}
    )
    summary = {
        "status": status,
        "requested": args.count,
        "completed": completed,
        "unique_assignment_signatures": completed,
        "unique_cif_hashes": completed,
        "eligible_topologies": len(rows),
        "productive_topologies": sum(value > 0 for value in topology_counts.values()),
        "mean_per_topology": completed / len(rows),
        "build_attempts": attempts,
        "successful_build_fraction": completed / attempts if attempts else 0.0,
        "strict_build_attempts": sum(int(row["strict_attempts"]) for row in rows),
        "strict_source_group_successes": sum(int(row["strict_exact"]) for row in rows),
        "relaxed_build_attempts": sum(int(row["relaxed_attempts"]) for row in rows),
        "p1_count": p1,
        "p1_fraction": p1 / completed if completed else 0.0,
        "raw_cartesian_positions_scanned": sum(int(row["raw_scanned"]) for row in rows),
        "active_seconds": float(meta(connection, "active_seconds", "0") or 0),
        "generation_phase": meta(connection, "generation_phase"),
        "strict_phase_completed_count": int(
            meta(connection, "strict_phase_completed_count", "0") or 0
        ),
        "topology_counts": topology_counts,
        "per_topology_count": per_topology_count,
        "topology_quota_complete": (
            not topology_shortfalls if per_topology_count is not None else None
        ),
        "topology_shortfalls": topology_shortfalls,
        "parked_topologies": sorted(row["name"] for row in rows if row["parked"]),
        "exhausted_topologies": sorted(row["name"] for row in rows if row["exhausted"]),
        "failure_counts": failure_counts,
        "settings": {
            "seed": config.seed,
            "start_phase": args.start_phase,
            "topology_mode": args.topology_mode,
            "per_topology_count": per_topology_count,
            "connectivity_policy": args.connectivity_policy,
            "workers": args.workers,
            "max_topology_slots": args.max_topology_slots,
            "max_output_atoms": args.max_output_atoms,
            "eligibility_symmetry": args.start_phase,
            "exclude_topology_file": (
                str(args.exclude_topology_file.expanduser().resolve())
                if args.exclude_topology_file is not None
                else None
            ),
            "worker_memory_limit_gib": WORKER_MEMORY_GIB,
            "worker_restart_jobs": RESTART_JOBS,
            "edge_conformer_filter": "all",
            "strategy": (
                "equal_per_topology_relaxed_only"
                if per_topology_count is not None and args.start_phase == "relaxed"
                else "equal_per_topology_strict_then_optional_relaxation"
                if per_topology_count is not None
                else "relaxed_only"
                if args.start_phase == "relaxed"
                else "global_strict_then_optional_relaxation"
            ),
            "strict_candidate_filter": "cheap_space_plus_lazy_exact_graph_symmetry",
            "relax_after_strict_exhaustion": args.relax_after_strict_exhaustion,
            "builder_symmetry_tolerance_angstrom": config.symmetry_tolerance,
            "classification_symprec_angstrom": SYMPREC,
        },
        "format": {
            "database": "generation.sqlite3",
            "cifs": "mofs/<shard>/single_linker_mof_<id>_<topology>.cif",
            "assignments": "samples.nodes_json and samples.edges_json",
        },
    }
    write_json(output / "generation_summary.json", summary)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.commit()
    if status == "complete":
        shutil.rmtree(output / ".staging", ignore_errors=True)
        shutil.rmtree(output / ".state", ignore_errors=True)
    return summary


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def topology_names_from_file(path: Path) -> set[str]:
    """Read topology names from JSON screening output or a text list."""

    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        names = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        if isinstance(payload, dict):
            names = payload.get("eligible_topologies")
        elif isinstance(payload, list):
            names = payload
        else:
            names = None
        if not isinstance(names, list):
            raise ValueError(
                f"topology file must be a JSON list, source_screening.json, "
                f"or newline list: {path}"
            )
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError(f"invalid topology name in {path}")
    return {name.strip() for name in names}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--count",
        type=positive,
        help=f"global target count (default: {DEFAULT_COUNT})",
    )
    target.add_argument(
        "--per-topology-count",
        type=positive,
        help="exact target for every eligible topology",
    )
    parser.add_argument("--workers", type=positive, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-topology-slots",
        type=positive,
        default=30,
        help="maximum expanded node-plus-edge positions per topology (default: 30)",
    )
    parser.add_argument(
        "--max-output-atoms",
        type=positive,
        default=0,
        help=(
            "maximum atoms in each stored explicit cell after connection sites "
            "are removed; 0 disables the limit (default: 0)"
        ),
    )
    parser.add_argument(
        "--topology-mode",
        choices=("binodal", "uninodal"),
        default="binodal",
        help=(
            "binodal: metal node + organic node; uninodal: metal node + "
            "required organic self-edge linker"
        ),
    )
    parser.add_argument(
        "--connectivity-policy",
        choices=("current", "legacy"),
        default="current",
        help=(
            "current requires validated 3D periodic connectivity; legacy is "
            "only for consistently resuming data created before that policy"
        ),
    )
    parser.add_argument(
        "--topology",
        action="append",
        default=[],
        help="limit generation to one eligible topology (repeatable)",
    )
    parser.add_argument(
        "--exclude-topology-file",
        type=Path,
        help=(
            "exclude names in a newline/JSON list or another dataset's "
            "source_screening.json"
        ),
    )
    parser.add_argument(
        "--start-phase",
        choices=("strict", "relaxed"),
        default="strict",
        help="start a new output in strict or relaxed generation",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "2m_mofs")
    parser.add_argument("--progress-interval", type=positive, default=30)
    parser.add_argument(
        "--relax-after-strict-exhaustion",
        action="store_true",
        help="open relaxed queues only after every strict queue is exhausted",
    )
    args = parser.parse_args()
    if args.count is None and args.per_topology_count is None:
        args.count = DEFAULT_COUNT
    return args


def main() -> None:
    warnings.filterwarnings("ignore")
    args = arguments()
    output = args.output.expanduser().resolve()
    database = output / "generation.sqlite3"
    if output.exists() and any(output.iterdir()) and not database.is_file():
        raise FileExistsError(f"nonempty non-resumable output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    library = (ROOT / "lib").resolve()
    manifest = output / "block_symmetry_manifest.csv"
    if database.is_file():
        if not manifest.is_file():
            raise FileNotFoundError(
                f"resumable output is missing its source manifest: {manifest}"
            )
        print(
            f"building_block_symmetry_manifest=resume_existing path={manifest}",
            flush=True,
        )
    else:
        manifest = write_symmetry_manifest(
            library,
            output,
            (ROOT / "generated/building_block_symmetry_cache").resolve(),
            args.workers,
        )
    config = core.GenerationConfig(
        manifest=manifest,
        topology_dir=(library / "topology").resolve(),
        output=output,
        count=args.count or args.per_topology_count,
        seed=args.seed,
        max_topology_slots=args.max_topology_slots,
        max_output_atoms=args.max_output_atoms,
        edge_conformer_filter="all",
        strict_symmetry=False,
        strict_prefilter=True,
        require_3d_connectivity=args.connectivity_policy == "current",
        attempt_batch_size=ATTEMPT_BATCH,
        worker_memory_limit_gib=WORKER_MEMORY_GIB,
    )
    pools, skipped_blocks = core.load_block_pools(config)
    topologies, skipped_topologies = core.load_eligible_topologies(
        config,
        pools,
        topology_mode=args.topology_mode,
        symmetry_mode=args.start_phase,
    )
    topologies = sorted(topologies, key=lambda item: item.name)
    excluded_topologies: set[str] = set()
    if args.exclude_topology_file is not None:
        excluded_topologies = topology_names_from_file(args.exclude_topology_file)
        topologies = [
            item for item in topologies if item.name not in excluded_topologies
        ]
        print(
            f"excluded_topologies={len(excluded_topologies)} "
            f"source={args.exclude_topology_file.expanduser().resolve()}",
            flush=True,
        )
    if args.topology:
        requested = set(args.topology)
        eligible_names = {item.name for item in topologies}
        missing = sorted(requested - eligible_names)
        if missing:
            raise RuntimeError(
                f"topologies are not eligible in {args.topology_mode} mode: "
                + ", ".join(missing)
            )
        topologies = [item for item in topologies if item.name in requested]
    if not topologies:
        raise RuntimeError(f"no eligible {args.topology_mode} topologies were found")
    if args.per_topology_count is not None:
        args.count = args.per_topology_count * len(topologies)
    config = replace(config, count=args.count)

    selected = Counter(block.kind for block in core.unique_library_blocks(pools))
    print(
        f"selected_blocks={dict(sorted(selected.items()))} "
        f"eligible_topologies={len(topologies)} target={args.count} "
        f"mean_target={args.count / len(topologies):.2f}",
        flush=True,
    )
    source_fingerprint = fingerprint(
        config,
        topologies,
        args.topology_mode,
        args.per_topology_count,
    )
    connection = connect(database)
    fresh = initialize(
        connection,
        source_fingerprint,
        args.count,
        args.relax_after_strict_exhaustion,
        args.start_phase,
        args.topology_mode,
        args.per_topology_count,
        args.connectivity_policy,
        args.max_output_atoms,
    )
    completed = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    phase = str(meta(connection, "generation_phase", "strict"))
    if phase not in {"strict", "relaxed"}:
        raise RuntimeError(f"unknown stored generation phase: {phase}")

    if completed < args.count:
        prepare_phase(
            connection,
            config,
            pools,
            topologies,
            output / ".state",
            args.workers,
            phase,
        )
    capacity = connection.execute(
        "SELECT COALESCE(SUM(direct_size),0),COALESCE(SUM(inserted_size),0) "
        "FROM topology_state"
    ).fetchone()
    print(
        f"generation_phase={phase} exact_candidate_capacity "
        f"direct={int(capacity[0])} inserted={int(capacity[1])} resume={not fresh}",
        flush=True,
    )
    write_json(
        output / "source_screening.json",
        {
            "selected_blocks": dict(selected),
            "skipped_blocks": dict(skipped_blocks),
            "skipped_topologies": dict(skipped_topologies),
            "eligible_topologies": [item.name for item in topologies],
            "topology_mode": args.topology_mode,
            "max_topology_slots": args.max_topology_slots,
            "max_output_atoms": args.max_output_atoms,
            "eligibility_symmetry": args.start_phase,
            "excluded_topologies": sorted(excluded_topologies),
            "exclude_topology_file": (
                str(args.exclude_topology_file.expanduser().resolve())
                if args.exclude_topology_file is not None
                else None
            ),
            "per_topology_count": args.per_topology_count,
            "connectivity_policy": args.connectivity_policy,
            "target_count": args.count,
            "source_fingerprint": source_fingerprint,
        },
    )

    completed, status = run_phase(connection, config, output, args, phase)
    if phase == "strict" and status in {"complete", "strict_exhausted"}:
        set_meta(connection, "strict_phase_completed_count", completed)
        connection.commit()
    if status == "strict_exhausted" and args.relax_after_strict_exhaustion:
        phase = "relaxed"
        set_meta(connection, "generation_phase", phase)
        connection.commit()
        print(
            f"strict_phase_exhausted={completed}/{args.count} "
            "starting_relaxed_phase=true",
            flush=True,
        )
        prepare_phase(
            connection,
            config,
            pools,
            topologies,
            output / ".state",
            args.workers,
            phase,
        )
        completed, status = run_phase(connection, config, output, args, phase)

    summary = finalize(connection, output, config, args, status)
    connection.close()
    print(
        f"generation_status={status} completed={completed}/{args.count} "
        f"attempts={summary['build_attempts']} output={output}",
        flush=True,
    )
    if completed != args.count:
        raise RuntimeError(
            f"generated {completed} of {args.count}; inspect {output / 'progress.json'}"
        )


if __name__ == "__main__":
    main()
