#!/usr/bin/env python3
"""Merge assignment- and CIF-unique supplemental MOFs into a finalized set."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from single_linker_mof_generator import write_results


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def update_saved_block_paths(
    row: dict[str, object],
    old_sample_id: str,
    new_sample_id: str,
) -> None:
    old_prefix = f"building_blocks/{old_sample_id}/"
    new_prefix = f"building_blocks/{new_sample_id}/"
    assignments = row["node_assignments"] + row["edge_assignments"]
    for assignment in assignments:
        saved_xyz = str(assignment["block"]["saved_xyz"])
        if not saved_xyz.startswith(old_prefix):
            raise ValueError(
                f"unexpected supplemental block path: {saved_xyz}"
            )
        assignment["block"]["saved_xyz"] = (
            new_prefix + saved_xyz.removeprefix(old_prefix)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary", type=Path)
    parser.add_argument("supplement", type=Path)
    parser.add_argument("--target-count", type=int, default=10_000)
    args = parser.parse_args()

    primary = args.primary.expanduser().resolve()
    supplement = args.supplement.expanduser().resolve()
    primary_rows = load_json(primary / "manifest.json")
    supplement_rows = load_json(supplement / "manifest.json")
    summary = load_json(primary / "generation_summary.json")
    needed = args.target_count - len(primary_rows)
    if needed < 0:
        raise ValueError("primary manifest already exceeds the target count")

    primary_ids = {row["sample_id"] for row in primary_rows}
    signatures = {row["assignment_signature"] for row in primary_rows}
    cif_hashes = {row["quality"]["cif_sha256"] for row in primary_rows}
    if (
        len(primary_ids) != len(primary_rows)
        or len(signatures) != len(primary_rows)
        or len(cif_hashes) != len(primary_rows)
    ):
        raise ValueError("primary manifest is not unique")

    selected = []
    overlap_signatures = 0
    overlap_hashes = 0
    if needed:
        for source_row in supplement_rows:
            signature = source_row["assignment_signature"]
            cif_hash = source_row["quality"]["cif_sha256"]
            if signature in signatures:
                overlap_signatures += 1
                continue
            if cif_hash in cif_hashes:
                overlap_hashes += 1
                continue
            source_cif = supplement / source_row["cif"]
            if file_sha256(source_cif) != cif_hash:
                raise ValueError(f"supplemental CIF hash mismatch: {source_cif}")
            selected.append(source_row)
            signatures.add(signature)
            cif_hashes.add(cif_hash)
            if len(selected) == needed:
                break
    if len(selected) != needed:
        raise RuntimeError(
            f"only {len(selected)} unique supplemental rows for {needed} slots"
        )

    created_paths: list[Path] = []
    merged_rows = list(primary_rows)
    try:
        for index, source_row in enumerate(
            selected,
            start=len(primary_rows) + 1,
        ):
            row = deepcopy(source_row)
            topology_name = row["topology"]["name"]
            old_sample_id = row["sample_id"]
            new_sample_id = f"single_linker_mof_{index:05d}_{topology_name}"
            if new_sample_id in primary_ids:
                raise ValueError(f"duplicate output sample id: {new_sample_id}")

            source_cif = supplement / row["cif"]
            destination_cif = primary / "mofs" / f"{new_sample_id}.cif"
            source_blocks = supplement / "building_blocks" / old_sample_id
            destination_blocks = primary / "building_blocks" / new_sample_id
            if destination_cif.exists() or destination_blocks.exists():
                raise FileExistsError(new_sample_id)
            shutil.copy2(source_cif, destination_cif)
            created_paths.append(destination_cif)
            shutil.copytree(source_blocks, destination_blocks)
            created_paths.append(destination_blocks)

            source_topology = supplement / row["topology"]["saved_cgd"]
            destination_topology = primary / row["topology"]["saved_cgd"]
            if not destination_topology.exists():
                shutil.copy2(source_topology, destination_topology)
                created_paths.append(destination_topology)

            row["sample_id"] = new_sample_id
            row["cif"] = str(destination_cif.relative_to(primary))
            update_saved_block_paths(row, old_sample_id, new_sample_id)
            merged_rows.append(row)
            primary_ids.add(new_sample_id)
    except Exception:
        for path in reversed(created_paths):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        raise

    eligible_topologies = sorted(
        set(summary.get("candidate_count_by_topology", {}))
        | {row["topology"]["name"] for row in merged_rows}
    )
    topology_counts = {
        name: sum(row["topology"]["name"] == name for row in merged_rows)
        for name in eligible_topologies
    }
    count_values = list(topology_counts.values())
    productive_count_values = [value for value in count_values if value]
    supplement_summary = summary.get("supplement")
    if needed:
        supplement_summary = {
            "source": str(supplement),
            "source_count": len(supplement_rows),
            "selected_count": len(selected),
            "assignment_overlaps_skipped_before_selection": (
                overlap_signatures
            ),
            "cif_hash_overlaps_skipped_before_selection": overlap_hashes,
        }
    summary.update(
        {
            "requested": args.target_count,
            "completed": len(merged_rows),
            "mode": "topology_first_without_replacement_plus_unique_supplement",
            "generation_seeds": sorted(
                {row["generation_seed"] for row in merged_rows}
            ),
            "supplement": supplement_summary,
            "eligible_topologies": len(topology_counts),
            "productive_topologies": len(productive_count_values),
            "final_count_by_topology": topology_counts,
            "final_count_statistics_all_eligible": {
                "mean": statistics.fmean(count_values),
                "median": statistics.median(count_values),
                "minimum": min(count_values),
                "maximum": max(count_values),
            },
            "final_count_statistics_productive": {
                "mean": statistics.fmean(productive_count_values),
                "median": statistics.median(productive_count_values),
                "minimum": min(productive_count_values),
                "maximum": max(productive_count_values),
            },
            "unique_assignment_signatures": len(
                {row["assignment_signature"] for row in merged_rows}
            ),
            "unique_cif_hashes": len(
                {row["quality"]["cif_sha256"] for row in merged_rows}
            ),
        }
    )
    write_results(primary, merged_rows, summary)
    print(
        f"merged={len(selected)} completed={len(merged_rows)} "
        f"assignment_overlaps_skipped={overlap_signatures} "
        f"hash_overlaps_skipped={overlap_hashes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
