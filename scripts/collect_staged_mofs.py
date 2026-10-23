#!/usr/bin/env python3
"""Collect already-accepted parallel MOF builds without rebuilding them."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "10k_mofs"
DEFAULT_TOPOLOGIES = PROJECT_ROOT.parent / "lib" / "topology"

NODE_PATTERN = re.compile(
    r"^#\s+Node\s+\d+,\s+Type\s+(\d+),\s+"
    r"BuildingBlock:\s+([^,]+),"
)
EDGE_PATTERN = re.compile(
    r"^#\s+Edge\s+\d+,\s+Type\s+\((\d+),\s*(\d+)\),\s+"
    r"(?:BuildingBlock:\s+([^,]+),|None)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_block_metadata(path: Path) -> dict[str, dict[str, object]]:
    fields = (
        "library_name",
        "source_key",
        "structure_id",
        "bb_kind",
        "output_path",
        "xyz_sha256",
        "num_atoms",
        "coordination_number",
        "port_point_group",
        "symmetry_basis",
        "graph_port_automorphism_order",
        "graph_supports_port_point_group",
    )
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            row["library_name"]: {field: row.get(field) for field in fields}
            for row in rows
        }


def comment_float(lines: list[str], prefix: str) -> float | None:
    for line in lines:
        if line.startswith(prefix):
            return float(line.split(":", 1)[1].strip())
    return None


def cif_value(lines: list[str], key: str) -> str | None:
    for line in lines:
        if line.startswith(key):
            fields = line.split(maxsplit=1)
            return fields[1].strip(" '\"") if len(fields) == 2 else None
    return None


def hill_formula(counts: Counter[str]) -> str:
    order = []
    if "C" in counts:
        order.append("C")
        if "H" in counts:
            order.append("H")
    order.extend(sorted(symbol for symbol in counts if symbol not in order))
    return " ".join(f"{symbol}{counts[symbol]}" for symbol in order)


def parse_accepted_cif(
    path: Path,
    blocks: dict[str, dict[str, object]],
) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    nodes: dict[int, str] = {}
    edges: dict[tuple[int, int], str] = {}
    for line in lines:
        node_match = NODE_PATTERN.match(line)
        if node_match:
            node_type = int(node_match.group(1))
            block_name = node_match.group(2).strip()
            if node_type in nodes and nodes[node_type] != block_name:
                raise ValueError(f"inconsistent node assignment in {path}")
            nodes[node_type] = block_name
            continue
        edge_match = EDGE_PATTERN.match(line)
        if edge_match and edge_match.group(3):
            edge_type = (int(edge_match.group(1)), int(edge_match.group(2)))
            block_name = edge_match.group(3).strip()
            if edge_type in edges and edges[edge_type] != block_name:
                raise ValueError(f"inconsistent edge assignment in {path}")
            edges[edge_type] = block_name

    if len(nodes) != 2 or len(edges) > 1:
        raise ValueError(f"not a single-linker assignment: {path}")
    missing = sorted((set(nodes.values()) | set(edges.values())) - blocks.keys())
    if missing:
        raise ValueError(f"unknown building blocks in {path}: {missing}")
    node_kinds = {str(blocks[name]["bb_kind"]) for name in nodes.values()}
    if node_kinds != {"metal_node", "organic_node"}:
        raise ValueError(f"node kinds are not metal+organic in {path}")

    try:
        atom_start = lines.index("_atom_type_partial_charge") + 1
    except ValueError as error:
        raise ValueError(f"missing atom table in {path}") from error
    elements: Counter[str] = Counter()
    for line in lines[atom_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("_", "loop_", "data_", "#")):
            break
        fields = stripped.split()
        if len(fields) < 6:
            break
        elements[fields[1]] += 1
    if not elements or "X" in elements or "C" not in elements:
        raise ValueError(f"invalid accepted atom table in {path}")

    cell_keys = (
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    )
    cell_parameters = [float(cif_value(lines, key)) for key in cell_keys]
    declared_number = cif_value(lines, "_symmetry_Int_Tables_number")
    geometry_loss = comment_float(
        lines,
        "# Geometry optimization loss:",
    )
    if geometry_loss is None:
        # Accept CIFs written before the loss metadata was renamed.
        geometry_loss = comment_float(lines, "# Geometry optimization objective:")
    if geometry_loss is None:
        geometry_loss = comment_float(lines, "# Relax obj value:")
    quality = {
        "num_atoms": sum(elements.values()),
        "formula": hill_formula(elements),
        "element_counts": dict(sorted(elements.items())),
        "cell_parameters": cell_parameters,
        "max_rmsd": comment_float(lines, "# Max RMSD:"),
        "mean_rmsd": comment_float(lines, "# Mean RMSD:"),
        "geometry_optimization_loss": geometry_loss,
        "relax_obj": geometry_loss,
        "minimum_atom_distance_angstrom": None,
        "declared_space_group_symbol": cif_value(
            lines, "_symmetry_space_group_name_H-M"
        ),
        "declared_space_group_number": (
            int(declared_number) if declared_number is not None else None
        ),
        "cif_sha256": sha256(path),
        "collection_validation": (
            "trusted: CIF was written only after the original parallel "
            "generation worker passed structural validation"
        ),
        "revalidated_during_collection": False,
    }
    return {
        "source": path,
        "topology_name": path.parent.name,
        "assignment_signature": path.stem,
        "node_assignments": [
            {
                "node_type": node_type,
                "block": blocks[block_name],
            }
            for node_type, block_name in sorted(nodes.items())
        ],
        "edge_assignments": [
            {
                "edge_type": list(edge_type),
                "block": blocks[block_name],
            }
            for edge_type, block_name in sorted(edges.items())
        ],
        "quality": quality,
    }


def round_robin(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["topology_name"])].append(record)
    for values in grouped.values():
        values.sort(key=lambda value: str(value["assignment_signature"]))
    result = []
    for index in range(max(map(len, grouped.values()))):
        for topology in sorted(grouped):
            if index < len(grouped[topology]):
                result.append(grouped[topology][index])
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "sample_id",
            "topology",
            "formula",
            "num_atoms",
            "max_rmsd",
            "cif_sha256",
            "cif",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            quality = row["quality"]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "topology": row["topology"]["name"],
                    "formula": quality["formula"],
                    "num_atoms": quality["num_atoms"],
                    "max_rmsd": quality["max_rmsd"],
                    "cif_sha256": quality["cif_sha256"],
                    "cif": row["cif"],
                }
            )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--topologies", type=Path, default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--requested", type=int, default=10_000)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    staging = output / ".topology_quota_staging"
    block_manifest = output / "block_symmetry_manifest.csv"
    mofs = output / "mofs"
    if not staging.is_dir() or not block_manifest.is_file():
        parser.error("staging directory or block symmetry manifest is missing")
    protected = [mofs, output / "manifest.json", output / "manifest.csv"]
    existing = [path for path in protected if path.exists()]
    if existing:
        parser.error(f"refusing to overwrite collected output: {existing}")

    blocks = load_block_metadata(block_manifest)
    staged_paths = sorted(staging.glob("*/*.cif"))
    if not staged_paths:
        parser.error("no staged CIFs found")
    records = round_robin(
        [parse_accepted_cif(path, blocks) for path in staged_paths]
    )
    signatures = [str(record["assignment_signature"]) for record in records]
    cif_hashes = [str(record["quality"]["cif_sha256"]) for record in records]
    if len(set(signatures)) != len(records):
        raise ValueError("duplicate assignment signatures in staging")
    if len(set(cif_hashes)) != len(records):
        raise ValueError("duplicate CIF hashes in staging")

    mofs.mkdir()
    topology_output = output / "topologies"
    topology_output.mkdir(exist_ok=False)
    topology_counts = Counter(
        str(record["topology_name"]) for record in records
    )
    for topology in topology_counts:
        source = args.topologies.expanduser().resolve() / f"{topology}.cgd"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, topology_output / source.name)

    rows = []
    for index, record in enumerate(records, start=1):
        topology = str(record["topology_name"])
        sample_id = f"single_linker_mof_{index:05d}_{topology}"
        destination = mofs / f"{sample_id}.cif"
        os.replace(record["source"], destination)
        rows.append(
            {
                "sample_id": sample_id,
                "cif": str(destination.relative_to(output)),
                "assignment_signature": record["assignment_signature"],
                "topology": {
                    "name": topology,
                    "source_cgd": str(
                        args.topologies.expanduser().resolve()
                        / f"{topology}.cgd"
                    ),
                    "saved_cgd": str(
                        (topology_output / f"{topology}.cgd").relative_to(output)
                    ),
                },
                "node_assignments": record["node_assignments"],
                "edge_assignments": record["edge_assignments"],
                "quality": record["quality"],
            }
        )

    summary = {
        "requested_original": args.requested,
        "completed": len(rows),
        "remaining_not_generated": max(0, args.requested - len(rows)),
        "stopped_by_user": True,
        "mode": "direct_collection_of_accepted_parallel_staging",
        "collection_revalidation": False,
        "productive_topologies": len(topology_counts),
        "count_by_topology": dict(sorted(topology_counts.items())),
        "unique_assignment_signatures": len(set(signatures)),
        "unique_cif_hashes": len(set(cif_hashes)),
        "duplicates_detected": 0,
        "contains_exactly_one_metal_and_one_organic_node": True,
        "maximum_optional_edge_assignments": 1,
    }
    atomic_json(output / "manifest.json", rows)
    atomic_json(output / "generation_summary.json", summary)
    write_csv(output / "manifest.csv", rows)
    shutil.rmtree(staging)
    print(
        f"collected={len(rows)} requested_original={args.requested} "
        f"topologies={len(topology_counts)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
