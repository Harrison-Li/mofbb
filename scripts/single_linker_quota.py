"""Balanced 10k generation: bounded candidate queues, quotas, and staging recovery.

This runner is separate from the bulk generator. It uses the shared core's
record types and validation rules so existing checkpoints remain readable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from itertools import islice, zip_longest
import math
import os
from pathlib import Path
import random
import resource
import shutil
import statistics
import time

from threadpoolctl import threadpool_limits

import single_linker_mof_generator as core


_CANDIDATE_WORKER_CONFIG: core.GenerationConfig | None = None
_CANDIDATE_WORKER_POOLS: (
    dict[tuple[str, int, str], list[core.BlockRecord]] | None
) = None


def configure_candidate_worker(
    config: core.GenerationConfig,
    pools: dict[tuple[str, int, str], list[core.BlockRecord]],
) -> None:
    """Initialize one parallel topology-candidate worker."""

    global _CANDIDATE_WORKER_CONFIG, _CANDIDATE_WORKER_POOLS
    _CANDIDATE_WORKER_CONFIG = config
    _CANDIDATE_WORKER_POOLS = pools
    core.configure_quiet_builder_logging()
    limit = config.worker_memory_limit_gib * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def enumerate_topology_candidates_job(
    topology: core.TopologyRecord,
) -> tuple[str, tuple[core.CandidateAssignment, ...], dict[str, object]]:
    """Enumerate one bounded topology queue in a worker process."""

    config, pools = _CANDIDATE_WORKER_CONFIG, _CANDIDATE_WORKER_POOLS
    if config is None or pools is None:
        raise RuntimeError("candidate worker was not initialized")
    started = time.monotonic()
    strict_prefilter = config.strict_symmetry and config.strict_prefilter
    node_stats: dict[str, int] = {}
    edge_stats: dict[str, int] = {}
    exact_symmetry_stats: dict[str, int | float] = {}
    # Small alignment problems benefit from processes rather than BLAS threads.
    with threadpool_limits(limits=1, user_api="blas"):
        if strict_prefilter:
            pools, node_stats = core.strict_node_compatible_pools(config, topology, pools)
            pools, edge_stats = core.strict_edge_compatible_pools(topology, pools)
        candidates = enumerate_topology_candidates(config, topology, pools)
        if strict_prefilter:
            candidates, exact_symmetry_stats = core.prefilter_candidate_queue(
                config, topology, candidates,
            )
    return topology.name, candidates, {
        "elapsed_seconds": time.monotonic() - started,
        "nodes": node_stats,
        "edges": edge_stats,
        "exact_symmetry": exact_symmetry_stats,
    }


def copy_selected_blocks(
    directory: Path,
    nodes: dict[int, core.BlockRecord],
    edges: dict[tuple[int, int], core.BlockRecord],
) -> tuple[dict[int, Path], dict[tuple[int, int], Path]]:
    directory.mkdir(parents=True)
    node_paths: dict[int, Path] = {}
    for node_type, record in nodes.items():
        target = directory / f"node_type_{node_type}.xyz"
        shutil.copy2(record.path, target)
        node_paths[node_type] = target
    edge_paths: dict[tuple[int, int], Path] = {}
    for edge_type, record in edges.items():
        label = "_".join(str(value) for value in edge_type)
        target = directory / f"edge_type_{label}.xyz"
        shutil.copy2(record.path, target)
        edge_paths[edge_type] = target
    return node_paths, edge_paths


def block_manifest(record: core.BlockRecord, saved_path: Path) -> dict[str, object]:
    data = asdict(record)
    data["path"] = str(record.path)
    data["saved_xyz"] = str(saved_path)
    return data


def enumerate_topology_candidates(
    config: core.GenerationConfig,
    topology: core.TopologyRecord,
    pools: dict[tuple[str, int, str], list[core.BlockRecord]],
) -> tuple[core.CandidateAssignment, ...]:
    """Build a bounded deterministic queue without materializing its product."""

    site_candidates: dict[
        tuple[int, str],
        list[tuple[core.BlockRecord, float]],
    ] = {}
    for site in topology.node_sites:
        for kind in ("metal_node", "organic_node"):
            if config.strict_symmetry and config.strict_prefilter:
                fitted = pools.get(
                    (kind, site.node_type, site.coordination, site.point_group),
                    [],
                )
                site_candidates[(site.node_type, kind)] = [
                    (item.block, item.rmsd) for item in fitted
                ]
                continue
            compatible = []
            for block in core.node_records_for_site(pools, kind, site):
                fit = core.screened_port_fit_rmsd(
                    site,
                    block,
                    config.max_rmsd,
                    config.accuracy,
                )
                if fit <= config.max_rmsd:
                    compatible.append((block, fit))
            site_candidates[(site.node_type, kind)] = compatible

    orientations = []
    for metal_site in topology.node_sites:
        for organic_site in topology.node_sites:
            if metal_site.node_type == organic_site.node_type:
                continue
            metal_blocks = tuple(
                site_candidates[(metal_site.node_type, "metal_node")]
            )
            organic_blocks = tuple(
                site_candidates[(organic_site.node_type, "organic_node")]
            )
            if metal_blocks and organic_blocks:
                orientations.append(
                    (
                        metal_site,
                        organic_site,
                        metal_blocks,
                        organic_blocks,
                    )
                )

    def permuted_indices(size: int, label: str):
        """Traverse a Cartesian index space bijectively in seeded order."""

        if size <= 0:
            return
        digest = hashlib.sha256(
            f"{config.seed}:{topology.name}:{label}".encode()
        ).digest()
        offset = int.from_bytes(digest[:16], "big") % size
        if size == 1:
            yield 0
            return
        step = int.from_bytes(digest[16:], "big") % size or 1
        while math.gcd(step, size) != 1:
            step += 1
            if step >= size:
                step = 1
        scan_limit = min(
            size,
            config.max_candidates_per_topology
            * config.candidate_scan_multiplier,
        )
        for index in range(scan_limit):
            yield (offset + index * step) % size

    def locate_orientation(
        global_index: int,
        sizes: list[int],
    ) -> tuple[
        tuple[
            core.NodeSite,
            core.NodeSite,
            tuple[tuple[core.BlockRecord, float], ...],
            tuple[tuple[core.BlockRecord, float], ...],
        ],
        int,
    ]:
        for orientation, size in zip(orientations, sizes, strict=True):
            if global_index < size:
                return orientation, global_index
            global_index -= size
        raise IndexError("candidate Cartesian index exceeds its spaces")

    strict_prefilter = config.strict_symmetry and config.strict_prefilter
    edge_options = tuple(
        (edge_type, block)
        for edge_type in topology.edge_types
        for block in pools.get(
            ("edge", edge_type, 2, core.EDGE_POINT_GROUP)
            if strict_prefilter else ("edge", 2, core.EDGE_POINT_GROUP),
            (),
        )
    )

    def order_key(candidate: core.CandidateAssignment) -> tuple[object, ...]:
        tie_breaker = hashlib.sha256(
            f"{config.seed}:{topology.name}:{candidate.signature}".encode()
        ).hexdigest()
        return (*candidate.score, tie_breaker)

    direct_cap = (
        config.direct_candidate_cap
        if config.direct_candidate_cap > 0 else config.max_candidates_per_topology
    )
    queues = []
    for label, options, cap in (
        ("direct", (None,), direct_cap),
        ("inserted", edge_options, config.max_candidates_per_topology),
    ):
        sizes = [
            len(metal_blocks) * len(organic_blocks) * len(options)
            for _, _, metal_blocks, organic_blocks in orientations
        ]
        candidates: dict[str, core.CandidateAssignment] = {}
        for global_index in permuted_indices(sum(sizes), label):
            if len(candidates) >= cap:
                break
            orientation, local_index = locate_orientation(global_index, sizes)
            metal_site, organic_site, metal_blocks, organic_blocks = orientation
            local_index, edge_index = divmod(local_index, len(options))
            metal_index, organic_index = divmod(local_index, len(organic_blocks))
            metal_block, metal_fit = metal_blocks[metal_index]
            organic_block, organic_fit = organic_blocks[organic_index]
            edge_option = options[edge_index]
            edges = {} if edge_option is None else {edge_option[0]: edge_option[1]}
            blocks = (metal_block, organic_block, *edges.values())
            if (
                len({block.sha256 for block in blocks}) != len(blocks)
                or len({block.structure_id for block in blocks}) != len(blocks)
                or not any(block.contains_carbon for block in blocks[1:])
            ):
                continue
            nodes = {
                metal_site.node_type: metal_block,
                organic_site.node_type: organic_block,
            }
            risk = core.assignment_prebuild_risk(
                topology, nodes, edges,
                maximum_rmsd=config.max_rmsd,
                accuracy=config.accuracy,
            )
            if risk is None:
                continue
            signature = core.candidate_assignment_signature(topology.name, nodes, edges)
            candidate = core.CandidateAssignment(
                signature=signature,
                nodes=tuple(sorted(nodes.items())),
                edges=tuple(sorted(edges.items())),
                score=(
                    risk,
                    max(metal_fit, organic_fit),
                    max(block.port_radius_span for block in nodes.values()),
                ),
            )
            if signature not in candidates or order_key(candidate) < order_key(
                candidates[signature]
            ):
                candidates[signature] = candidate
        queues.append(sorted(candidates.values(), key=order_key))

    if strict_prefilter:
        return tuple(sorted((*queues[0], *queues[1]), key=order_key))
    interleaved = (
        candidate for pair in zip_longest(*queues)
        for candidate in pair if candidate is not None
    )
    return tuple(islice(interleaved, config.max_candidates_per_topology))


def build_topology_quota(job: dict[str, object]) -> dict[str, object]:
    """Consume one topology's candidate queue without replacement."""

    config = job["config"]
    topology_record = job["topology"]
    candidates = job["candidates"]
    start = int(job["start"])
    target = int(job["target"])
    attempt_limit = int(job["attempt_limit"])
    pass_index = int(job["pass_index"])
    staging = Path(job["staging"])
    staging.mkdir(parents=True, exist_ok=True)
    core.configure_quiet_builder_logging()
    successes = []
    failure_types: Counter[str] = Counter()
    failure_examples = []
    attempted = 0
    cursor = start

    with threadpool_limits(limits=1, user_api="blas"):
        try:
            topology = core.cached_topology(str(topology_record.cgd_path))
        except Exception as error:
            return {
                "topology": topology_record.name,
                "pass_index": pass_index,
                "start": start,
                "next_cursor": start,
                "attempted": 0,
                "requested": target,
                "successes": [],
                "failure_types": {type(error).__name__: 1},
                "failure_examples": [
                    {
                        "candidate_index": None,
                        "error_type": type(error).__name__,
                        "error": str(error)[:1000],
                    }
                ],
                "topology_error": True,
            }

        stop = min(len(candidates), start + attempt_limit)
        for candidate_index in range(start, stop):
            if len(successes) >= target:
                break
            candidate = candidates[candidate_index]
            attempted += 1
            cursor = candidate_index + 1
            cif_path = staging / f"{candidate.signature}.cif"
            nodes = dict(candidate.nodes)
            edges = dict(candidate.edges)
            try:
                quality = core.build_and_validate(
                    config,
                    topology_record,
                    {key: block.path for key, block in nodes.items()},
                    {key: block.path for key, block in edges.items()},
                    cif_path,
                    topology=topology,
                )
                quality["cif_sha256"] = core.file_sha256(cif_path)
                successes.append(
                    {
                        "candidate_index": candidate_index,
                        "candidate": candidate,
                        "cif_path": str(cif_path),
                        "quality": quality,
                    }
                )
            except Exception as error:
                cif_path.unlink(missing_ok=True)
                error_type = type(error).__name__
                failure_types[error_type] += 1
                if len(failure_examples) < 20:
                    failure_examples.append(
                        {
                            "candidate_index": candidate_index,
                            "error_type": error_type,
                            "error": str(error)[:1000],
                        }
                    )

    return {
        "topology": topology_record.name,
        "pass_index": pass_index,
        "start": start,
        "next_cursor": cursor,
        "attempted": attempted,
        "requested": target,
        "successes": successes,
        "failure_types": dict(failure_types),
        "failure_examples": failure_examples,
        "topology_error": False,
    }


def balanced_extra_targets(
    current_counts: dict[str, int],
    active_names: list[str],
    needed: int,
) -> dict[str, int]:
    """Water-fill extra quotas so productive topologies stay balanced."""

    targets: Counter[str] = Counter()
    projected = dict(current_counts)
    for _ in range(needed):
        name = min(
            active_names,
            key=lambda value: (projected[value], value),
        )
        targets[name] += 1
        projected[name] += 1
    return dict(targets)


def round_robin_successes(
    successes: dict[str, list[dict[str, object]]],
) -> list[tuple[str, dict[str, object]]]:
    """Interleave topology outputs instead of grouping one net at a time."""

    names = sorted(successes)
    return [
        (name, success)
        for row in zip_longest(*(successes[name] for name in names))
        for name, success in zip(names, row)
        if success is not None
    ]


def generate_topology_balanced(
    config: core.GenerationConfig,
    *,
    workers: int = core.DEFAULT_GENERATION_WORKERS,
    max_passes: int = 30,
    resume: bool = False,
) -> list[dict]:
    """Generate a balanced, without-replacement set topology by topology."""

    if (
        config.count < 1
        or workers < 1
        or max_passes < 1
        or config.max_candidates_per_topology < 1
        or config.edge_conformer_filter not in {"all", "non_c1"}
        or config.candidate_scan_multiplier < 1
        or config.attempt_batch_size < 1
        or config.adaptive_min_attempts < 1
        or not 0.0 <= config.adaptive_minimum_yield <= 1.0
        or config.adaptive_confidence_z <= 0.0
        or config.worker_memory_limit_gib < 1
        or config.symmetry_tolerance <= 0.0
    ):
        raise ValueError(
            "generation counts, workers, and limits must be positive; "
            "edge_conformer_filter must be 'all' or 'non_c1'; "
            "adaptive_minimum_yield must be between zero and one"
        )
    output = config.output
    protected = [
        output / "mofs",
        output / "building_blocks",
        output / "manifest.json",
        output / "manifest.csv",
        output / "generation_summary.json",
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing results: {existing}")

    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    staging = output / ".topology_quota_staging"
    if staging.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite staging: {staging}")
    if resume and not staging.is_dir():
        raise FileNotFoundError(f"resume staging directory not found: {staging}")
    staging.mkdir(exist_ok=True)
    core.configure_quiet_builder_logging()
    # Point-group and small alignment calculations do not benefit from a large
    # BLAS team.  Limiting this serial phase also prevents it from contending
    # with the process-parallel candidate phase that follows.
    with threadpool_limits(limits=1, user_api="blas"):
        pools, skipped_blocks = core.load_block_pools(config)
        selected_block_counts = Counter(
            record.kind for record in core.unique_library_blocks(pools)
        )
        print(
            f"selected_blocks={dict(sorted(selected_block_counts.items()))} "
            f"edge_conformer_filter={config.edge_conformer_filter}",
            flush=True,
        )
        if not selected_block_counts["edge"]:
            raise RuntimeError(
                "the selected symmetry cache contains no usable edges"
            )
        if config.strict_symmetry and config.strict_prefilter:
            strict_edges = core.strict_edge_conformer_pool(pools)
            print(
                f"site_routed_edge_pool={len(strict_edges)} "
                f"including_c1_and_directional=true",
                flush=True,
            )
        topologies, skipped_topologies = core.load_eligible_topologies(config, pools)
    topologies = sorted(topologies, key=lambda item: item.name)
    if not topologies:
        raise RuntimeError("no eligible two-node topologies were found")

    candidate_queues: dict[str, tuple[core.CandidateAssignment, ...]] = {}
    candidate_preparation_stats: dict[str, dict[str, object]] = {}
    topology_by_name = {topology.name: topology for topology in topologies}
    with ProcessPoolExecutor(
        max_workers=min(workers, len(topologies)),
        initializer=configure_candidate_worker,
        initargs=(config, pools),
    ) as executor:
        futures = {
            executor.submit(enumerate_topology_candidates_job, topology): topology
            for topology in topologies
        }
        for index, future in enumerate(as_completed(futures), start=1):
            topology = futures[future]
            try:
                name, candidates, preparation_stats = future.result()
            except Exception as error:
                name = topology.name
                candidates = ()
                preparation_stats = {}
                skipped_topologies[
                    f"candidate_enumeration_{type(error).__name__}"
                ] += 1
                print(
                    f"candidate_topology_error name={name} "
                    f"error={type(error).__name__}:{str(error)[:500]}",
                    flush=True,
                )
            candidate_queues[name] = candidates
            candidate_preparation_stats[name] = preparation_stats
            exact_stats = preparation_stats.get("exact_symmetry", {})
            print(
                f"candidate_topology={index}/{len(topologies)} "
                f"name={name} unique_assignments={len(candidates)} "
                f"symmetry_pruned={exact_stats.get('rejected_candidates', 0)}",
                flush=True,
            )

    preparation_elapsed_seconds = time.monotonic() - started
    build_phase_started = time.monotonic()

    base_quota, remainder = divmod(config.count, len(topologies))
    quota_order = [topology.name for topology in topologies]
    random.Random(config.seed).shuffle(quota_order)
    initial_quotas = {
        name: base_quota + int(index < remainder)
        for index, name in enumerate(quota_order)
    }
    print(
        f"topology_first_plan={len(topologies)} target={config.count} "
        f"base_quota={base_quota} remainder={remainder} "
        f"mean_quota={config.count / len(topologies):.6f}",
        flush=True,
    )
    queue_fingerprints = {
        name: core.candidate_queue_fingerprint(candidates)
        for name, candidates in candidate_queues.items()
    }
    checkpoint_path = staging / "resume_checkpoint.jsonl"

    cursors = {name: 0 for name in topology_by_name}
    successes: dict[str, list[dict[str, object]]] = {
        name: [] for name in topology_by_name
    }
    seen_signatures: set[str] = set()
    seen_cif_hashes: set[str] = set()
    attempted_counts: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()
    failure_examples: list[dict[str, object]] = []
    duplicates = Counter()
    quarantined_topologies: set[str] = set()
    retired_low_yield: dict[str, dict[str, float | int]] = {}
    low_yield_fallbacks: list[dict[str, object]] = []
    pass_summaries = []
    resumed_count = 0
    resume_start_pass = 1

    def update_low_yield_status(name: str, pass_index: int) -> bool:
        retirement = core.low_yield_retirement(
            len(successes[name]),
            attempted_counts[name],
            minimum_attempts=config.adaptive_min_attempts,
            minimum_yield=config.adaptive_minimum_yield,
            confidence_z=config.adaptive_confidence_z,
        )
        if retirement is None:
            return False
        if name not in retired_low_yield:
            retired_low_yield[name] = {
                **retirement,
                "retired_after_pass": pass_index,
                "cursor": cursors[name],
                "queue_size": len(candidate_queues[name]),
            }
            print(
                f"topology_queue_retired name={name} pass={pass_index} "
                f"successes={len(successes[name])} "
                f"attempts={attempted_counts[name]} "
                f"observed_yield={retirement['observed_yield']:.6f} "
                f"upper_bound={retirement['yield_upper_bound']:.6f} "
                f"minimum_yield={config.adaptive_minimum_yield:.6f}",
                flush=True,
            )
        return True

    def productive_topology_names(pass_index: int) -> list[str]:
        normal = [
            name
            for name in topology_by_name
            if successes[name]
            and name not in quarantined_topologies
            and name not in retired_low_yield
            and cursors[name] < len(candidate_queues[name])
        ]
        if normal:
            productive_cap = max(32, workers * 4)
            normal.sort(
                key=lambda name: (
                    len(successes[name]) / max(1, attempted_counts[name]),
                    len(successes[name]),
                    name,
                ),
                reverse=True,
            )
            return normal[:productive_cap]

        # Retired queues are a reserve, not a hard exclusion: if they are the
        # only source left, probe only the statistically strongest ones so a
        # strict yield cutoff cannot make an otherwise achievable target fail.
        fallback = [
            name
            for name in retired_low_yield
            if successes[name]
            and name not in quarantined_topologies
            and cursors[name] < len(candidate_queues[name])
        ]
        fallback.sort(
            key=lambda name: (
                core.wilson_upper_bound(
                    len(successes[name]),
                    attempted_counts[name],
                    config.adaptive_confidence_z,
                ),
                len(successes[name]),
                name,
            ),
            reverse=True,
        )
        selected = fallback[:workers]
        if selected:
            event = {
                "pass_index": pass_index,
                "topologies": selected,
                "reason": "all_nonretired_productive_queues_exhausted",
            }
            low_yield_fallbacks.append(event)
            print(
                f"low_yield_fallback pass={pass_index} "
                f"topologies={','.join(selected)}",
                flush=True,
            )
        return selected

    if resume:
        checkpoint_records = core.read_resume_checkpoint(checkpoint_path)
        valid_jobs: dict[tuple[int, str], dict[str, object]] = {}
        checkpoint_successes: dict[
            tuple[str, str], dict[str, object]
        ] = {}
        expected_validation = core.validation_fingerprint(config)
        for record in checkpoint_records:
            try:
                name = str(record["topology"])
                topology = topology_by_name[name]
                topology_stat = topology.cgd_path.stat()
                valid = (
                    int(record["version"]) == core.RESUME_CHECKPOINT_VERSION
                    and record["validation_fingerprint"] == expected_validation
                    and record["queue_fingerprint"] == queue_fingerprints[name]
                    and int(record["topology_size"]) == topology_stat.st_size
                    and int(record["topology_mtime_ns"])
                    == topology_stat.st_mtime_ns
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not valid:
                continue
            pass_index = int(record["pass_index"])
            valid_jobs[(pass_index, name)] = record
            for saved in record.get("successes", []):
                checkpoint_successes[(name, str(saved["signature"]))] = saved

        for (pass_index, name), record in sorted(valid_jobs.items()):
            attempted_counts[name] += int(record["attempted"])
            cursors[name] = max(cursors[name], int(record["next_cursor"]))
            resume_start_pass = max(resume_start_pass, pass_index + 1)

        recovered_by_name: dict[
            str,
            list[tuple[int, dict[str, object]]],
        ] = defaultdict(list)
        recovery_jobs = []
        staged_count = 0
        checkpoint_loaded = 0

        def add_recovered(
            name: str,
            candidate_index: int,
            candidate: core.CandidateAssignment,
            cif_path: Path,
            quality: dict[str, object],
        ) -> None:
            if candidate.signature in seen_signatures:
                raise ValueError(
                    f"duplicate staged assignment signature: {candidate.signature}"
                )
            cif_hash = str(quality["cif_sha256"])
            if cif_hash in seen_cif_hashes:
                raise ValueError(f"duplicate staged CIF hash: {cif_hash}")
            seen_signatures.add(candidate.signature)
            seen_cif_hashes.add(cif_hash)
            quality = {**quality, "resumed_from_staging": True}
            recovered_by_name[name].append(
                (
                    candidate_index,
                    {
                        "candidate_index": candidate_index,
                        "candidate": candidate,
                        "cif_path": str(cif_path),
                        "quality": quality,
                    },
                )
            )

        for name, candidates in candidate_queues.items():
            by_signature = {
                candidate.signature: (index, candidate)
                for index, candidate in enumerate(candidates)
            }
            for cif_path in sorted((staging / name).glob("*.cif")):
                candidate_entry = by_signature.get(cif_path.stem)
                if candidate_entry is None:
                    raise ValueError(
                        f"staged signature is absent from regenerated queue: "
                        f"{cif_path}"
                    )
                candidate_index, candidate = candidate_entry
                staged_count += 1
                saved = checkpoint_successes.get((name, candidate.signature))
                trusted = False
                if saved is not None and not config.resume_revalidate:
                    try:
                        cif_stat = cif_path.stat()
                        quality = dict(saved["quality"])
                        trusted = (
                            int(saved["candidate_index"]) == candidate_index
                            and saved["cif_name"] == cif_path.name
                            and int(saved["cif_size"]) == cif_stat.st_size
                            and int(saved["cif_mtime_ns"])
                            == cif_stat.st_mtime_ns
                            and len(str(quality["cif_sha256"])) == 64
                            and quality.get("connectivity_policy_version")
                            == core.CONNECTIVITY_POLICY_VERSION
                            and quality.get("single_linker_policy_version")
                            == core.SINGLE_LINKER_POLICY_VERSION
                            and quality.get("num_linker_types") == 1
                        )
                    except (KeyError, TypeError, ValueError):
                        trusted = False
                if trusted:
                    add_recovered(
                        name,
                        candidate_index,
                        candidate,
                        cif_path,
                        quality,
                    )
                    checkpoint_loaded += 1
                else:
                    recovery_jobs.append(
                        (name, candidate_index, candidate, cif_path)
                    )

        if not staged_count:
            raise RuntimeError("resume staging contains no CIF files")
        print(
            f"resume_recovery_plan={staged_count} "
            f"checkpoint_loaded={checkpoint_loaded} "
            f"revalidate={len(recovery_jobs)}",
            flush=True,
        )
        if recovery_jobs:
            with ProcessPoolExecutor(
                max_workers=min(workers, len(recovery_jobs)),
                initializer=core.configure_generation_worker,
                initargs=(config.worker_memory_limit_gib,),
            ) as executor:
                validated = 0
                batch_size = max(32, workers * 4)
                for batch_start in range(0, len(recovery_jobs), batch_size):
                    batch = recovery_jobs[batch_start : batch_start + batch_size]
                    futures = {
                        executor.submit(core.validate_staged_cif, config, cif_path): (
                            name,
                            candidate_index,
                            candidate,
                            cif_path,
                        )
                        for name, candidate_index, candidate, cif_path in batch
                    }
                    for future in as_completed(futures):
                        name, candidate_index, candidate, cif_path = futures[future]
                        add_recovered(
                            name,
                            candidate_index,
                            candidate,
                            cif_path,
                            future.result(),
                        )
                        validated += 1
                        if validated % 250 == 0 or validated == len(recovery_jobs):
                            print(
                                f"resume_validated={validated}/{len(recovery_jobs)}",
                                flush=True,
                            )

        for name in candidate_queues:
            recovered = recovered_by_name[name]
            recovered.sort(key=lambda value: value[0])
            successes[name].extend(value for _, value in recovered)
            if recovered:
                cursors[name] = max(cursors[name], recovered[-1][0] + 1)

        resumed_count = sum(len(values) for values in successes.values())
        prior_failures = max(0, sum(attempted_counts.values()) - resumed_count)
        failure_types["prior_run_rejected_candidate"] += prior_failures
        print(
            f"resume_loaded={resumed_count}/{config.count} "
            f"resume_start_pass={resume_start_pass} "
            f"prior_attempted={sum(attempted_counts.values())}",
            flush=True,
        )
        for name in topology_by_name:
            update_low_yield_status(name, max(0, resume_start_pass - 1))

    def accept_result(result: dict[str, object]) -> int:
        name = str(result["topology"])
        cursors[name] = max(cursors[name], int(result["next_cursor"]))
        attempted_counts[name] += int(result["attempted"])
        failure_types.update(result["failure_types"])
        for failure in result["failure_examples"]:
            if len(failure_examples) < 1000:
                failure_examples.append(
                    {
                        **failure,
                        "topology": name,
                        "pass_index": result["pass_index"],
                    }
                )
        accepted_successes = []
        for success in result["successes"]:
            candidate = success["candidate"]
            cif_path = Path(success["cif_path"])
            if candidate.signature in seen_signatures:
                duplicates["assignment"] += 1
                cif_path.unlink(missing_ok=True)
                continue
            cif_hash = success["quality"]["cif_sha256"]
            if cif_hash in seen_cif_hashes:
                duplicates["cif"] += 1
                cif_path.unlink(missing_ok=True)
                continue
            seen_signatures.add(candidate.signature)
            seen_cif_hashes.add(cif_hash)
            successes[name].append(success)
            accepted_successes.append(success)
        core.append_resume_checkpoint(
            checkpoint_path,
            config,
            topology_by_name[name],
            queue_fingerprints[name],
            result,
            accepted_successes,
        )
        return len(accepted_successes)

    if resumed_count:
        active_names = productive_topology_names(resume_start_pass)
        if resumed_count < config.count and not active_names:
            raise RuntimeError("no productive candidate queues remain for resume")
        targets = balanced_extra_targets(
            {name: len(successes[name]) for name in topology_by_name},
            active_names,
            config.count - resumed_count,
        )
    else:
        targets = initial_quotas
    generation_executor = ProcessPoolExecutor(
        max_workers=min(workers, len(topology_by_name)),
        initializer=core.configure_generation_worker,
        initargs=(config.worker_memory_limit_gib,),
    )
    for pass_index in range(resume_start_pass, max_passes + 1):
        jobs = []
        for name, target in sorted(targets.items()):
            remaining_candidates = len(candidate_queues[name]) - cursors[name]
            if target <= 0 or remaining_candidates <= 0:
                continue
            jobs.append(
                {
                    "config": config,
                    "topology": topology_by_name[name],
                    "candidates": candidate_queues[name],
                    "start": cursors[name],
                    "target": target,
                    "attempt_limit": min(
                        remaining_candidates,
                        config.attempt_batch_size,
                    ),
                    "pass_index": pass_index,
                    "staging": str(staging / name),
                }
            )
        if not jobs:
            break

        pass_attempted = 0
        pass_accepted = 0
        futures = {
            generation_executor.submit(build_topology_quota, job): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            name = str(job["topology"].name)
            try:
                result = future.result()
            except Exception as error:
                quarantined_topologies.add(name)
                result = {
                    "topology": name,
                    "pass_index": pass_index,
                    "next_cursor": len(candidate_queues[name]),
                    "attempted": 0,
                    "successes": [],
                    "failure_types": {type(error).__name__: 1},
                    "failure_examples": [
                        {
                            "candidate_index": None,
                            "error_type": type(error).__name__,
                            "error": str(error)[:1000],
                        }
                    ],
                }
            accepted = accept_result(result)
            update_low_yield_status(name, pass_index)
            pass_attempted += int(result["attempted"])
            pass_accepted += accepted
            print(
                f"quota_pass={pass_index} topology={name} "
                f"accepted={accepted}/{job['target']} "
                f"total={len(successes[name])} "
                f"attempted={result['attempted']} "
                f"cursor={cursors[name]}/{len(candidate_queues[name])}",
                flush=True,
            )

        total = sum(len(values) for values in successes.values())
        pass_summaries.append(
            {
                "pass_index": pass_index,
                "jobs": len(jobs),
                "attempted": pass_attempted,
                "accepted": pass_accepted,
                "cumulative": total,
                "retired_low_yield_topologies": len(retired_low_yield),
            }
        )
        print(
            f"quota_pass_complete={pass_index} attempted={pass_attempted} "
            f"accepted={pass_accepted} total={total}/{config.count}",
            flush=True,
        )
        if total >= config.count:
            break

        productive = productive_topology_names(pass_index + 1)
        if not productive:
            break
        needed = config.count - total
        targets = balanced_extra_targets(
            {name: len(successes[name]) for name in topology_by_name},
            productive,
            needed,
        )
    generation_executor.shutdown()
    build_elapsed_seconds = time.monotonic() - build_phase_started

    finalization_started = time.monotonic()
    ordered_successes = round_robin_successes(successes)[: config.count]
    mofs_dir = output / "mofs"
    blocks_dir = output / "building_blocks"
    topology_dir = output / "topologies"
    for directory in (mofs_dir, blocks_dir, topology_dir):
        directory.mkdir()
    used_topologies = sorted({name for name, _ in ordered_successes})
    for name in used_topologies:
        shutil.copy2(
            topology_by_name[name].cgd_path,
            topology_dir / f"{name}.cgd",
        )

    rows = []
    for index, (name, success) in enumerate(ordered_successes, start=1):
        topology = topology_by_name[name]
        candidate = success["candidate"]
        sample_id = f"single_linker_mof_{index:05d}_{name}"
        final_cif = mofs_dir / f"{sample_id}.cif"
        os.replace(success["cif_path"], final_cif)
        final_blocks = blocks_dir / sample_id
        nodes = dict(candidate.nodes)
        edges = dict(candidate.edges)
        node_paths, edge_paths = copy_selected_blocks(
            final_blocks,
            nodes,
            edges,
        )
        row = {
            "sample_id": sample_id,
            "cif": str(final_cif.relative_to(output)),
            "assignment_signature": candidate.signature,
            "generation_seed": config.seed,
            "topology": {
                "name": topology.name,
                "cgd_path": str(topology.cgd_path),
                "saved_cgd": str(
                    (topology_dir / f"{name}.cgd").relative_to(output)
                ),
                "space_group": topology.space_group,
                "num_nodes": topology.num_nodes,
                "num_edges": topology.num_edges,
                "node_sites": [asdict(site) for site in topology.node_sites],
                "edge_types": [list(value) for value in topology.edge_types],
            },
            "node_assignments": [
                {
                    "node_type": node_type,
                    "block": block_manifest(
                        block,
                        final_blocks.relative_to(output)
                        / node_paths[node_type].name,
                    ),
                }
                for node_type, block in nodes.items()
            ],
            "edge_assignments": [
                {
                    "edge_type": list(edge_type),
                    "block": block_manifest(
                        block,
                        final_blocks.relative_to(output)
                        / edge_paths[edge_type].name,
                    ),
                }
                for edge_type, block in edges.items()
            ],
            "quality": success["quality"],
        }
        rows.append(row)

    final_counts = {
        name: sum(row["topology"]["name"] == name for row in rows)
        for name in topology_by_name
    }
    count_values = list(final_counts.values())
    finalization_elapsed_seconds = time.monotonic() - finalization_started
    preparation_totals = {
        section: dict(
            sum(
                (
                    Counter(stats.get(section, {}))
                    for stats in candidate_preparation_stats.values()
                ),
                Counter(),
            )
        )
        for section in ("nodes", "edges", "exact_symmetry")
    }
    summary = {
        "requested": config.count,
        "completed": len(rows),
        "seed": config.seed,
        "mode": "topology_first_without_replacement",
        "resumed_from_staging": resumed_count,
        "elapsed_seconds": time.monotonic() - started,
        "phase_elapsed_seconds": {
            "block_loading_topology_screening_candidate_enumeration": (
                preparation_elapsed_seconds
            ),
            "building_and_testing": build_elapsed_seconds,
            "finalization": finalization_elapsed_seconds,
        },
        "candidate_preparation_totals": preparation_totals,
        "source_manifest": str(config.manifest),
        "topology_directory": str(config.topology_dir),
        "eligible_topologies": len(topologies),
        "productive_topologies": sum(value > 0 for value in count_values),
        "initial_mean_quota": config.count / len(topologies),
        "initial_quota_by_topology": dict(sorted(initial_quotas.items())),
        "candidate_count_by_topology": {
            name: len(candidate_queues[name]) for name in sorted(candidate_queues)
        },
        "attempted_count_by_topology": dict(sorted(attempted_counts.items())),
        "final_count_by_topology": dict(sorted(final_counts.items())),
        "final_count_statistics_all_eligible": {
            "mean": statistics.fmean(count_values),
            "median": statistics.median(count_values),
            "minimum": min(count_values),
            "maximum": max(count_values),
        },
        "unique_assignment_signatures": len(
            {row["assignment_signature"] for row in rows}
        ),
        "unique_cif_hashes": len(
            {row["quality"]["cif_sha256"] for row in rows}
        ),
        "duplicates_removed": dict(duplicates),
        "quarantined_topologies": sorted(quarantined_topologies),
        "retired_low_yield_topologies": dict(sorted(retired_low_yield.items())),
        "low_yield_fallbacks": low_yield_fallbacks,
        "failed_attempts": sum(failure_types.values()),
        "failure_types": dict(failure_types),
        "failure_examples": failure_examples,
        "skipped_blocks": dict(skipped_blocks),
        "skipped_topologies": dict(skipped_topologies),
        "quota_passes": pass_summaries,
        "settings": {
            **asdict(config),
            "manifest": str(config.manifest),
            "topology_dir": str(config.topology_dir),
            "output": str(config.output),
            "workers": workers,
            "max_passes": max_passes,
            "selection_rule": (
                "one metal node + one organic node + optional CN2/D*h edge; "
                "route by site geometry and chemical port compatibility; "
                "strict mode requires crystallographic-site compatibility, "
                "otherwise allow independent P1 placement; sample seeded "
                "Cartesian queues without replacement, deduplicate, and balance "
                "topology quotas; redistribute shortfalls to productive queues "
                "and reserve low-Wilson-yield queues for exhaustion fallback"
            ),
        },
    }
    core.write_results(output, rows, summary)
    shutil.rmtree(staging)
    if len(rows) != config.count:
        raise RuntimeError(
            f"generated {len(rows)} of {config.count} requested unique MOFs; "
            f"see {output / 'generation_summary.json'}"
        )
    return rows
