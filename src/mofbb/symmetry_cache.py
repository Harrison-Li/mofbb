"""Persistent symmetry classification for MOF building blocks.

The cache deliberately separates three concepts: chemical graph symmetry,
the symmetry of one set of Cartesian coordinates, and crystallographic-site
compatibility.  Every connected, structurally valid block is retained in the
catalog.  Its graph action on the ``X`` connection points, coordinate-derived
point groups, and intrinsic linker-end equivalence are annotations consumed by
the topology-aware generator; none is a global admission requirement.  This is
important because a completely asymmetric block is valid on a general Wyckoff
position even when it cannot occupy a special position.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, NamedTuple

from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from threadpoolctl import threadpool_limits

from mofbb.compose.building_block import BuildingBlock
from mofbb.graph_symmetry import (
    analyze_port_graph_symmetry,
    point_group_port_action,
)


POINT_GROUP_TOLERANCE = 0.3
POINT_GROUP_EIGEN_TOLERANCE = 0.01
POINT_GROUP_MATRIX_TOLERANCE = 0.1
MAX_NODE_ATOMS = 120
MAX_EDGE_ATOMS = 80
MAX_NODE_COORDINATION = 12

CACHE_SCHEMA_VERSION = 5
CLASSIFIER_VERSION = 5
CACHE_MANIFEST_NAME = "block_symmetry_manifest.csv"
CACHE_METADATA_NAME = "metadata.json"
GRAPH_SYMMETRY_BASIS = "element_and_bond_labeled_graph_automorphism"

_CLASSIFICATION_THREAD_LIMIT = None

SYMMETRY_MANIFEST_FIELDS = (
    # Stable fields consumed by the existing generators.
    "library_name",
    "source_key",
    "structure_id",
    "bb_kind",
    "output_path",
    "xyz_sha256",
    "num_atoms",
    "coordination_number",
    "port_point_group",
    "molecular_point_group",
    "chirality",
    "linker_end_equivalence",
    "classification_status",
    "point_group_tolerance",
    "connection_anchor_symbols",
    # Graph-symmetry provenance and induced action on the X ports.
    "symmetry_basis",
    "graph_connected",
    "graph_automorphism_order",
    "graph_port_automorphism_order",
    "graph_port_automorphism_generators",
    "graph_port_orbits",
    # Coordinate diagnostics.  These do not participate in cache admission.
    "conformer_point_group",
    "port_point_group_action_order",
    "port_point_group_permutations",
    "port_point_group_max_mapping_residual",
    "graph_supports_port_point_group",
    "point_group_diagnostic_status",
    "classification_error",
)


class LibraryCandidate(NamedTuple):
    """One manifest row that passes the generators' size/shape filters."""

    name: str
    kind: str
    path: str
    coordination: int
    num_atoms: int
    expected_sha256: str
    structure_id: str


@dataclass(frozen=True)
class CacheResult:
    """Result of ensuring or rebuilding a persistent symmetry cache."""

    status: str
    cache: Path
    manifest: Path
    metadata_path: Path
    metadata: dict[str, object]

    @property
    def cache_status(self) -> str:
        """Return the legacy ``hit``/``miss`` status spelling."""

        return self.status

    @property
    def manifest_path(self) -> Path:
        """Alias useful to callers that prefer an explicit path suffix."""

        return self.manifest


def file_sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _bool_field(value: bool) -> str:
    return "true" if value else "false"


def _base_row(candidate: LibraryCandidate, actual_sha256: str) -> dict[str, object]:
    return {
        "library_name": candidate.name,
        "source_key": candidate.path,
        "structure_id": candidate.structure_id,
        "bb_kind": candidate.kind,
        "output_path": candidate.path,
        "xyz_sha256": actual_sha256,
        "num_atoms": candidate.num_atoms,
        "coordination_number": candidate.coordination,
        "port_point_group": "analysis_failed",
        "molecular_point_group": "analysis_failed",
        "chirality": "unclassified",
        "linker_end_equivalence": "unclassified" if candidate.kind == "edge" else "na",
        "classification_status": "malformed_block",
        "point_group_tolerance": POINT_GROUP_TOLERANCE,
        "connection_anchor_symbols": "[]",
        "symmetry_basis": GRAPH_SYMMETRY_BASIS,
        "graph_connected": "unclassified",
        "graph_automorphism_order": 0,
        "graph_port_automorphism_order": 0,
        "graph_port_automorphism_generators": "[]",
        "graph_port_orbits": "[]",
        "conformer_point_group": "analysis_failed",
        "port_point_group_action_order": 0,
        "port_point_group_permutations": "[]",
        "port_point_group_max_mapping_residual": "",
        "graph_supports_port_point_group": "unclassified",
        "point_group_diagnostic_status": "not_evaluated",
        "classification_error": "",
    }


def _coerce_candidate(
    item: LibraryCandidate | tuple[str, str, str, int, int, str, str],
) -> LibraryCandidate:
    if isinstance(item, LibraryCandidate):
        return item
    return LibraryCandidate(*item)


def classify_block(
    item: LibraryCandidate | tuple[str, str, str, int, int, str, str],
) -> dict[str, object]:
    """Classify a candidate by graph symmetry and attach geometry diagnostics.

    File-integrity and analysis failures propagate to the caller.
    """

    candidate = _coerce_candidate(item)
    path = Path(candidate.path)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != candidate.expected_sha256:
        raise ValueError(
            f"library hash mismatch for {path}: expected "
            f"{candidate.expected_sha256}, found {actual_sha256}"
        )

    row = _base_row(candidate, actual_sha256)
    block = BuildingBlock(path)
    row["connection_anchor_symbols"] = _compact_json(
        block.connection_point_anchor_symbols
    )

    if block.n_connection_points != candidate.coordination:
        row["classification_status"] = "coordination_mismatch"
        row["classification_error"] = (
            f"manifest coordination {candidate.coordination} differs from "
            f"the {block.n_connection_points} X atoms in {path}"
        )
        return row

    graph = analyze_port_graph_symmetry(block)

    row.update(
        {
            "graph_connected": _bool_field(graph.connected),
            "graph_automorphism_order": int(graph.graph_automorphism_order),
            "graph_port_automorphism_order": int(graph.order),
            "graph_port_automorphism_generators": _compact_json(graph.generators),
            "graph_port_orbits": _compact_json(graph.orbits),
            "linker_end_equivalence": (
                "equivalent" if graph.endpoint_equivalent else "inequivalent"
            )
            if candidate.kind == "edge"
            else "na",
        }
    )
    if not graph.connected:
        row["classification_status"] = "disconnected_graph"
        row["classification_error"] = "molecular graph is disconnected"
    else:
        row["classification_status"] = "complete"

    # Coordinate diagnostics are required for asymmetric blocks too: the
    # generator uses them to route C1 motifs to general positions and to avoid
    # expensive special-position checks that cannot succeed.
    if row["classification_status"] != "complete":
        return row

    port_analyzer = PointGroupAnalyzer(
        Molecule(
            ["He"] * block.n_connection_points,
            block.connection_points,
        ),
        tolerance=POINT_GROUP_TOLERANCE,
        eigen_tolerance=POINT_GROUP_EIGEN_TOLERANCE,
        matrix_tolerance=POINT_GROUP_MATRIX_TOLERANCE,
    )
    row["port_point_group"] = port_analyzer.sch_symbol
    port_action = point_group_port_action(port_analyzer, tolerance=float("inf"))
    row["port_point_group_action_order"] = len(port_action.permutations)
    row["port_point_group_permutations"] = _compact_json(port_action.permutations)
    row["port_point_group_max_mapping_residual"] = (
        f"{port_action.max_residual:.12g}"
    )
    row["graph_supports_port_point_group"] = _bool_field(
        graph.supports_all(port_action.permutations)
    )

    symbols = [
        "He" if symbol == "X" else symbol
        for symbol in block.atoms.get_chemical_symbols()
    ]
    conformer_analyzer = PointGroupAnalyzer(
        Molecule(symbols, block.atoms.positions),
        tolerance=POINT_GROUP_TOLERANCE,
        eigen_tolerance=POINT_GROUP_EIGEN_TOLERANCE,
        matrix_tolerance=POINT_GROUP_MATRIX_TOLERANCE,
    )
    conformer_group = conformer_analyzer.sch_symbol
    row["molecular_point_group"] = conformer_group
    row["conformer_point_group"] = conformer_group
    row["point_group_diagnostic_status"] = "complete"
    return row


def library_candidates(library: Path) -> list[LibraryCandidate]:
    """Return candidates using the same filters as both generation scripts."""

    library = Path(library).resolve()
    manifest = library / "building_blocks_manifest.csv"
    candidates: list[LibraryCandidate] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            kind = row["kind"]
            coordination = int(row["physical_cn"])
            logical_coordination = int(row["logical_cn"])
            num_atoms = int(row["num_atoms"])
            if row["logical_kind"] != kind or coordination != logical_coordination:
                continue
            if json.loads(row["logical_port_sizes"]) != [1] * coordination:
                continue
            if kind == "edge":
                if coordination != 2 or num_atoms > MAX_EDGE_ATOMS:
                    continue
            elif (
                kind not in {"metal_node", "organic_node"}
                or coordination < 3
                or coordination > MAX_NODE_COORDINATION
                or num_atoms > MAX_NODE_ATOMS
            ):
                continue
            candidates.append(
                LibraryCandidate(
                    row["name"],
                    kind,
                    str(library / row["path"]),
                    coordination,
                    num_atoms,
                    row["sha256"],
                    row["source_refcode"] or row["name"],
                )
            )
    return candidates


def cache_identity(
    library: Path,
    candidates: Iterable[
        LibraryCandidate | tuple[str, str, str, int, int, str, str]
    ],
) -> dict[str, object]:
    """Describe every input and setting that affects classification."""

    library = Path(library).resolve()
    candidates = [_coerce_candidate(item) for item in candidates]
    manifest_path = library / "building_blocks_manifest.csv"
    library_digest = hashlib.sha256()
    library_digest.update(file_sha256(manifest_path).encode())
    for candidate in candidates:
        path = Path(candidate.path)
        actual_sha256 = file_sha256(path)
        if actual_sha256 != candidate.expected_sha256:
            raise ValueError(
                f"library hash mismatch for {path}: expected "
                f"{candidate.expected_sha256}, found {actual_sha256}"
            )
        library_digest.update(str(path.relative_to(library)).encode())
        library_digest.update(b"\0")
        library_digest.update(actual_sha256.encode())

    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "library_root": str(library),
        "library_fingerprint": library_digest.hexdigest(),
        "candidate_count": len(candidates),
        "classification_settings": {
            "symmetry_basis": GRAPH_SYMMETRY_BASIS,
            "vertex_labels": ["element", "connection_point_role"],
            "edge_labels": ["bond_type"],
            "stereochemistry_encoded": False,
            "cache_admission": {
                "edge": "connected_two_port_graph_with_site_compatibility_deferred",
                "node": "connected_graph_with_site_compatibility_deferred",
            },
            "edge_endpoint_exchange_is_diagnostic_only": True,
            "point_groups_are_diagnostic_only": True,
            "point_group_tolerance": POINT_GROUP_TOLERANCE,
            "point_group_eigen_tolerance": POINT_GROUP_EIGEN_TOLERANCE,
            "point_group_matrix_tolerance": POINT_GROUP_MATRIX_TOLERANCE,
            "max_node_atoms": MAX_NODE_ATOMS,
            "max_edge_atoms": MAX_EDGE_ATOMS,
            "max_node_coordination": MAX_NODE_COORDINATION,
        },
    }


def _is_accepted(row: dict[str, object]) -> bool:
    """Return the kind-aware cache admission decision for one row."""

    if (
        row.get("classification_status") != "complete"
        or row.get("graph_connected") != "true"
    ):
        return False

    if int(row["graph_port_automorphism_order"]) < 1:
        return False

    if row.get("bb_kind") == "edge":
        return int(row["coordination_number"]) == 2

    return True


def _rejection_reason(row: dict[str, object]) -> str:
    status = str(row.get("classification_status", "missing_status"))
    if status != "complete":
        return status
    if row.get("graph_connected") != "true":
        return "disconnected_graph"
    if int(row["graph_port_automorphism_order"]) < 1:
        return "invalid_graph_port_automorphism_order"
    if row.get("bb_kind") == "edge":
        if int(row["coordination_number"]) != 2:
            return "invalid_edge_coordination"
        return "not_accepted"
    return "not_accepted"


def _cache_statistics(
    classified: list[dict[str, object]],
    accepted: list[dict[str, object]],
) -> dict[str, object]:
    candidate_by_kind = Counter(str(row["bb_kind"]) for row in classified)
    complete_by_kind = Counter(
        str(row["bb_kind"])
        for row in classified
        if row.get("classification_status") == "complete"
    )
    accepted_by_kind = Counter(str(row["bb_kind"]) for row in accepted)
    rejected_by_kind = candidate_by_kind - accepted_by_kind
    rejection_counts = Counter(
        _rejection_reason(row) for row in classified if not _is_accepted(row)
    )

    def valid_group(value: object) -> bool:
        return value not in {
            None,
            "",
            "analysis_failed",
            "not_evaluated",
            "unclassified",
        }

    accepted_conformer_c1 = [
        row for row in accepted if row.get("conformer_point_group") == "C1"
    ]
    accepted_mismatches = [
        row
        for row in accepted
        if valid_group(row.get("port_point_group"))
        and valid_group(row.get("conformer_point_group"))
        and row["port_point_group"] != row["conformer_point_group"]
    ]
    accepted_edges = [row for row in accepted if row.get("bb_kind") == "edge"]
    intrinsic_exchange_edges = [
        row
        for row in accepted_edges
        if row.get("linker_end_equivalence") == "equivalent"
    ]
    directional_edges = [
        row
        for row in accepted_edges
        if row.get("linker_end_equivalence") == "inequivalent"
    ]
    conformational_d_infinity_h_edges = [
        row
        for row in accepted_edges
        if row.get("port_point_group") == "D*h"
        and row.get("conformer_point_group") == "D*h"
    ]

    kinds = sorted(candidate_by_kind)
    counts_by_kind = {
        kind: {
            "candidates": candidate_by_kind[kind],
            "classification_complete": complete_by_kind[kind],
            "accepted": accepted_by_kind[kind],
            "rejected": rejected_by_kind[kind],
            "accepted_conformer_c1": sum(
                row["bb_kind"] == kind for row in accepted_conformer_c1
            ),
            "accepted_port_conformer_point_group_mismatch": sum(
                row["bb_kind"] == kind for row in accepted_mismatches
            ),
        }
        for kind in kinds
    }
    support_counts = Counter(
        str(row.get("graph_supports_port_point_group", "unclassified"))
        for row in accepted
    )
    candidate_counts = {kind: candidate_by_kind[kind] for kind in kinds}
    complete_counts = {kind: complete_by_kind[kind] for kind in kinds}
    accepted_counts = {kind: accepted_by_kind[kind] for kind in kinds}
    rejected_counts = {kind: rejected_by_kind[kind] for kind in kinds}
    return {
        "classified_count": len(classified),
        "classification_complete_count": sum(complete_by_kind.values()),
        "accepted_count": len(accepted),
        "rejected_count": len(classified) - len(accepted),
        "candidate_counts_by_kind": candidate_counts,
        "classification_complete_counts_by_kind": complete_counts,
        "accepted_counts_by_kind": accepted_counts,
        "rejected_counts_by_kind": rejected_counts,
        "counts_by_kind": counts_by_kind,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "accepted_conformer_c1_count": len(accepted_conformer_c1),
        "accepted_port_conformer_point_group_mismatch_count": len(
            accepted_mismatches
        ),
        "accepted_graph_supports_port_point_group_counts": dict(
            sorted(support_counts.items())
        ),
        "accepted_edge_count": len(accepted_edges),
        "accepted_intrinsic_endpoint_exchange_edge_count": len(
            intrinsic_exchange_edges
        ),
        "accepted_directional_edge_count": len(directional_edges),
        "accepted_conformational_d_infinity_h_edge_count": len(
            conformational_d_infinity_h_edges
        ),
    }


def load_valid_cache(
    cache: Path,
    identity: dict[str, object],
) -> dict[str, object] | None:
    """Load metadata when both cache files match the requested identity."""

    cache = Path(cache)
    manifest = cache / CACHE_MANIFEST_NAME
    metadata_path = cache / CACHE_METADATA_NAME
    if not manifest.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        return None
    if any(metadata.get(key) != value for key, value in identity.items()):
        return None
    if not all(
        type(metadata.get(key)) is int
        for key in ("classified_count", "accepted_count", "rejected_count")
    ):
        return None
    if metadata["classified_count"] != identity["candidate_count"]:
        return None
    if metadata["accepted_count"] + metadata["rejected_count"] != metadata[
        "classified_count"
    ]:
        return None
    if metadata.get("manifest_sha256") != file_sha256(manifest):
        return None
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SYMMETRY_MANIFEST_FIELDS:
            return None
        if sum(1 for _ in reader) != metadata["accepted_count"]:
            return None
    return metadata


def write_cache(
    cache: Path,
    identity: dict[str, object],
    classified: list[dict[str, object]],
    accepted: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Atomically write the accepted manifest and its validating metadata."""

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    if len(classified) != identity.get("candidate_count"):
        raise ValueError(
            "classified row count does not match the cache candidate identity"
        )
    if accepted is None:
        accepted = [row for row in classified if _is_accepted(row)]
    if any(not _is_accepted(row) for row in accepted):
        raise ValueError("accepted cache rows must satisfy symmetry admission")

    manifest = cache / CACHE_MANIFEST_NAME
    metadata_path = cache / CACHE_METADATA_NAME
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=cache,
        prefix=f".{CACHE_MANIFEST_NAME}.",
        delete=False,
    ) as handle:
        temporary_manifest = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=SYMMETRY_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(accepted)
        handle.flush()
        os.fsync(handle.fileno())

    metadata = {
        **identity,
        **_cache_statistics(classified, accepted),
        "manifest_sha256": file_sha256(temporary_manifest),
    }
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=cache,
        prefix=f".{CACHE_METADATA_NAME}.",
        delete=False,
    ) as handle:
        temporary_metadata = Path(handle.name)
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temporary_manifest, manifest)
    os.replace(temporary_metadata, metadata_path)
    return metadata


def _classify_candidates(
    candidates: list[LibraryCandidate], workers: int
) -> list[dict[str, object]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        with threadpool_limits(limits=1, user_api="blas"):
            return [classify_block(candidate) for candidate in candidates]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_configure_classification_worker,
    ) as executor:
        return list(executor.map(classify_block, candidates, chunksize=20))


def _configure_classification_worker() -> None:
    """Use process concurrency instead of nested 64-thread BLAS teams."""

    global _CLASSIFICATION_THREAD_LIMIT
    _CLASSIFICATION_THREAD_LIMIT = threadpool_limits(limits=1, user_api="blas")


def rebuild_cache(
    library: Path,
    cache: Path,
    workers: int = 1,
) -> CacheResult:
    """Reclassify the full filtered library and atomically replace the cache."""

    library = Path(library).resolve()
    cache = Path(cache).expanduser().resolve()
    candidates = library_candidates(library)
    identity = cache_identity(library, candidates)
    classified = _classify_candidates(candidates, workers)
    accepted = [row for row in classified if _is_accepted(row)]
    metadata = write_cache(cache, identity, classified, accepted)
    return CacheResult(
        status="miss",
        cache=cache,
        manifest=cache / CACHE_MANIFEST_NAME,
        metadata_path=cache / CACHE_METADATA_NAME,
        metadata=metadata,
    )


def ensure_cache(
    library: Path,
    cache: Path,
    workers: int = 1,
    force_rebuild: bool = False,
) -> CacheResult:
    """Return a valid cache, rebuilding it when absent, stale, or forced."""

    if workers < 1:
        raise ValueError("workers must be positive")
    library = Path(library).resolve()
    cache = Path(cache).expanduser().resolve()
    candidates = library_candidates(library)
    identity = cache_identity(library, candidates)
    if not force_rebuild:
        metadata = load_valid_cache(cache, identity)
        if metadata is not None:
            return CacheResult(
                status="hit",
                cache=cache,
                manifest=cache / CACHE_MANIFEST_NAME,
                metadata_path=cache / CACHE_METADATA_NAME,
                metadata=metadata,
            )

    classified = _classify_candidates(candidates, workers)
    accepted = [row for row in classified if _is_accepted(row)]
    metadata = write_cache(cache, identity, classified, accepted)
    return CacheResult(
        status="miss",
        cache=cache,
        manifest=cache / CACHE_MANIFEST_NAME,
        metadata_path=cache / CACHE_METADATA_NAME,
        metadata=metadata,
    )


def copy_cache_manifest(
    cache: Path | CacheResult,
    output: Path,
    filename: str = CACHE_MANIFEST_NAME,
) -> Path:
    """Atomically copy a validated cache manifest into an output directory."""

    source = (
        cache.manifest
        if isinstance(cache, CacheResult)
        else Path(cache) / CACHE_MANIFEST_NAME
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    with source.open("rb") as source_handle, tempfile.NamedTemporaryFile(
        "wb",
        dir=output,
        prefix=f".{filename}.",
        delete=False,
    ) as destination_handle:
        temporary = Path(destination_handle.name)
        shutil.copyfileobj(source_handle, destination_handle)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    os.replace(temporary, destination)
    return destination


def write_symmetry_manifest(
    library: Path,
    output: Path,
    cache: Path,
    workers: int,
    force_rebuild: bool = False,
) -> Path:
    """Ensure the shared cache, copy its manifest, and print CLI statistics."""

    result = ensure_cache(
        library,
        cache,
        workers=workers,
        force_rebuild=force_rebuild,
    )
    manifest = copy_cache_manifest(result, output)
    print(
        f"building_block_symmetry_cache={result.status} "
        f"classified_blocks={result.metadata['classified_count']} "
        f"site_routable_blocks={result.metadata['accepted_count']} "
        f"cache={result.cache}",
        flush=True,
    )
    return manifest


__all__ = [
    "CACHE_MANIFEST_NAME",
    "CACHE_METADATA_NAME",
    "CACHE_SCHEMA_VERSION",
    "CLASSIFIER_VERSION",
    "GRAPH_SYMMETRY_BASIS",
    "MAX_EDGE_ATOMS",
    "MAX_NODE_ATOMS",
    "MAX_NODE_COORDINATION",
    "POINT_GROUP_EIGEN_TOLERANCE",
    "POINT_GROUP_MATRIX_TOLERANCE",
    "POINT_GROUP_TOLERANCE",
    "SYMMETRY_MANIFEST_FIELDS",
    "CacheResult",
    "LibraryCandidate",
    "cache_identity",
    "classify_block",
    "copy_cache_manifest",
    "ensure_cache",
    "file_sha256",
    "library_candidates",
    "load_valid_cache",
    "rebuild_cache",
    "write_cache",
    "write_symmetry_manifest",
]
