#!/usr/bin/env python3
"""Build MOFs from grouped decomposed BBs and match the redecomposed graphs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import warnings

from mofbb.compose import Builder, BuildingBlock, Topology
from mofbb.compose.log import logger
from mofbb.decompose import NodeEdgeDecomposer
from mofbb.matching import (
    BuildingBlockGraph,
    compare_building_blocks,
    match_building_block_multisets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "examples" / "node-edge-decomposed"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "decomposed_bb_roundtrip"


@dataclass(frozen=True)
class CaseInputs:
    topology: Topology
    blocks: list[BuildingBlock | None]
    expected_instances: list[BuildingBlockGraph]
    expected_types: list[BuildingBlockGraph]


def renamed_graph(graph: BuildingBlockGraph, name: str) -> BuildingBlockGraph:
    return BuildingBlockGraph(
        name=name,
        kind=graph.kind,
        symbols=graph.symbols,
        bonds=graph.bonds,
    )


def unique_connectivity_types(
    graphs: list[BuildingBlockGraph],
) -> list[BuildingBlockGraph]:
    representatives: list[BuildingBlockGraph] = []
    for graph in graphs:
        if any(
            compare_building_blocks(
                representative,
                graph,
                require_kind=False,
                require_bond_types=False,
            ).matched
            for representative in representatives
        ):
            continue
        representatives.append(graph)
    return representatives


def load_case_inputs(decomposition_dir: Path) -> CaseInputs:
    reconstruction = json.loads(
        (decomposition_dir / "reconstruction_map.json").read_text(
            encoding="utf-8"
        )
    )
    topology = Topology(decomposition_dir / reconstruction["topology"]["path"])
    slots = sorted(reconstruction["slots"], key=lambda item: item["slot_id"])
    node_indices = [int(index) for index in topology.node_indices]
    edge_indices = [int(index) for index in topology.edge_indices]

    if all("topology_slot" in slot for slot in slots):
        target_indices = []
        for slot in slots:
            target = slot["topology_slot"]
            candidates = (
                node_indices if target["kind"] == "node" else edge_indices
            )
            target_indices.append(candidates[int(target["index"])])
    else:
        if len(slots) != len(node_indices):
            raise ValueError(
                f"{decomposition_dir} has {len(slots)} BB slots but its topology "
                f"has {len(node_indices)} node slots."
            )
        target_indices = node_indices

    group_records = {
        str(group["bb_id"]): group for group in reconstruction["bb_groups"]
    }
    group_blocks: dict[str, BuildingBlock] = {}
    group_graphs: dict[str, BuildingBlockGraph] = {}
    for bb_id, group in group_records.items():
        block = BuildingBlock(decomposition_dir / str(group["path"]))
        group_blocks[bb_id] = block
        group_graphs[bb_id] = BuildingBlockGraph.from_building_block(
            block,
            kind=str(group["kind"]),
            name=bb_id,
        )

    blocks: list[BuildingBlock | None] = [None] * topology.n_slots
    expected_instances = []
    for target_index, slot in zip(target_indices, slots, strict=True):
        bb_id = str(slot["bb_id"])
        if blocks[target_index] is not None:
            raise ValueError(f"Topology slot {target_index} is assigned twice.")
        blocks[target_index] = group_blocks[bb_id].copy()
        expected_instances.append(
            renamed_graph(
                group_graphs[bb_id],
                f"source_slot_{int(slot['slot_id'])}:{bb_id}",
            )
        )

    missing_nodes = [index for index in node_indices if blocks[index] is None]
    if missing_nodes:
        raise ValueError(f"Topology node slots are unassigned: {missing_nodes}.")

    return CaseInputs(
        topology=topology,
        blocks=blocks,
        expected_instances=expected_instances,
        expected_types=unique_connectivity_types(list(group_graphs.values())),
    )


def recovered_graphs(result: object) -> list[BuildingBlockGraph]:
    graphs = []
    for fragment in result.fragments:
        graph = BuildingBlockGraph.from_fragment(fragment)
        graphs.append(
            BuildingBlockGraph(
                name=graph.name,
                kind=fragment.logical_kind or fragment.kind,
                symbols=graph.symbols,
                bonds=graph.bonds,
            )
        )
    return graphs


def serialize_match(result: object) -> dict[str, object]:
    return {
        "reference_count": result.reference_count,
        "candidate_count": result.candidate_count,
        "matched_count": result.matched_count,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "exact": result.exact,
        "matched_pairs": [list(pair) for pair in result.matched_pairs],
    }


def run_case(
    decomposition_dir: Path,
    output: Path,
    accuracy: int,
) -> dict[str, object]:
    case_dir = output / "cases" / decomposition_dir.name
    case_dir.mkdir(parents=True)
    rebuilt_path = case_dir / "rebuilt.cif"
    redecomposition_dir = case_dir / "redecomposition"
    stage = "load_inputs"
    record: dict[str, object] = {
        "case": decomposition_dir.name,
        "source": str(decomposition_dir),
        "status": "failed",
        "failure_stage": None,
        "error": None,
        "expected_instance_count": None,
        "expected_type_count": None,
        "source_kind_counts": None,
        "composed_atoms": None,
        "recovered_instance_count": None,
        "recovered_type_count": None,
        "recovered_kind_counts": None,
        "instance_connectivity": None,
        "type_connectivity": None,
        "role_aware_instance_connectivity": None,
        "exact": False,
    }

    try:
        inputs = load_case_inputs(decomposition_dir)
        record["expected_instance_count"] = len(inputs.expected_instances)
        record["expected_type_count"] = len(inputs.expected_types)
        record["source_kind_counts"] = dict(
            sorted(Counter(graph.kind for graph in inputs.expected_instances).items())
        )

        stage = "composition"
        framework = Builder().build(
            inputs.topology,
            inputs.blocks,
            accuracy=accuracy,
        )
        framework.write_cif(rebuilt_path)
        if not rebuilt_path.is_file():
            raise RuntimeError("Framework CIF was not written.")
        record["composed_atoms"] = len(framework.atoms)

        stage = "redecomposition"
        decomposition = NodeEdgeDecomposer(
            rebuilt_path,
            cleanup=False,
            solvent_mode="none",
        ).decompose()
        decomposition.write(redecomposition_dir)

        stage = "matching"
        candidates = recovered_graphs(decomposition)
        candidate_types = unique_connectivity_types(candidates)
        instance_match = match_building_block_multisets(
            inputs.expected_instances,
            candidates,
            require_kind=False,
            require_bond_types=False,
        )
        type_match = match_building_block_multisets(
            inputs.expected_types,
            candidate_types,
            require_kind=False,
            require_bond_types=False,
        )
        role_aware_match = match_building_block_multisets(
            inputs.expected_instances,
            candidates,
            require_kind=True,
            require_bond_types=False,
        )

        record.update(
            {
                "status": "completed",
                "recovered_instance_count": len(candidates),
                "recovered_type_count": len(candidate_types),
                "recovered_kind_counts": dict(
                    sorted(Counter(graph.kind for graph in candidates).items())
                ),
                "instance_connectivity": serialize_match(instance_match),
                "type_connectivity": serialize_match(type_match),
                "role_aware_instance_connectivity": serialize_match(
                    role_aware_match
                ),
                "exact": instance_match.exact and type_match.exact,
            }
        )
    except Exception as error:
        record["failure_stage"] = stage
        record["error"] = (
            f"{type(error).__name__}: {' '.join(str(error).split())[:2000]}"
        )

    (case_dir / "result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return record


def rate(count: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "count": count,
        "denominator": denominator,
        "fraction": count / denominator if denominator else None,
    }


def aggregate_matches(
    records: list[dict[str, object]],
    field: str,
    expected_count_field: str,
    candidate_count_field: str,
) -> dict[str, object]:
    reference_count = sum(
        int(record[expected_count_field] or 0) for record in records
    )
    candidate_count = sum(
        int(record[candidate_count_field] or 0) for record in records
    )
    matched_count = sum(
        int(record[field]["matched_count"])
        for record in records
        if record[field] is not None
    )
    precision = matched_count / candidate_count if candidate_count else 0.0
    recall = matched_count / reference_count if reference_count else 0.0
    return {
        "reference_count": reference_count,
        "candidate_count": candidate_count,
        "matched_count": matched_count,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    }


def build_summary(
    records: list[dict[str, object]],
    accuracy: int,
) -> dict[str, object]:
    total = len(records)
    completed_records = [
        record for record in records if record["status"] == "completed"
    ]
    completed = len(completed_records)
    exact = sum(bool(record["exact"]) for record in records)
    role_exact = sum(
        bool(
            record["role_aware_instance_connectivity"]
            and record["role_aware_instance_connectivity"]["exact"]
        )
        for record in records
    )
    failures = Counter(
        str(record["failure_stage"])
        for record in records
        if record["status"] != "completed"
    )
    return {
        "configuration": {
            "accuracy": accuracy,
            "decomposer": (
                "NodeEdgeDecomposer(cleanup=False, solvent_mode='none')"
            ),
            "node_labels": "element symbols including X",
            "edge_labels": "ignored",
            "coordinates": "ignored",
            "atom_order": "ignored",
            "bb_kind": "ignored for primary matching",
            "matching": "maximum one-to-one labelled-graph isomorphism",
            "source_bbs": "grouped representatives from bbs/",
        },
        "cases": total,
        "completed": rate(completed, total),
        "exact_connectivity_roundtrips": rate(exact, total),
        "exact_conditional_on_completed": rate(exact, completed),
        "role_aware_exact_roundtrips": rate(role_exact, total),
        "instance_connectivity": aggregate_matches(
            records,
            "instance_connectivity",
            "expected_instance_count",
            "recovered_instance_count",
        ),
        "instance_connectivity_conditional_on_completed": aggregate_matches(
            completed_records,
            "instance_connectivity",
            "expected_instance_count",
            "recovered_instance_count",
        ),
        "type_connectivity": aggregate_matches(
            records,
            "type_connectivity",
            "expected_type_count",
            "recovered_type_count",
        ),
        "type_connectivity_conditional_on_completed": aggregate_matches(
            completed_records,
            "type_connectivity",
            "expected_type_count",
            "recovered_type_count",
        ),
        "failure_stages": dict(sorted(failures.items())),
    }


def write_report(
    output: Path,
    summary: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    lines = [
        "# Decomposed-BB connectivity round-trip",
        "",
        "| Case | Status | Expected instances | Recovered instances | Matched | Exact |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        match = record["instance_connectivity"] or {}
        lines.append(
            "| {case} | {status} | {expected} | {recovered} | {matched} | {exact} |".format(
                case=record["case"],
                status=record["status"],
                expected=record["expected_instance_count"] or 0,
                recovered=record["recovered_instance_count"] or 0,
                matched=match.get("matched_count", 0),
                exact="yes" if record["exact"] else "no",
            )
        )

    exact = summary["exact_connectivity_roundtrips"]
    conditional_exact = summary["exact_conditional_on_completed"]
    instance = summary["instance_connectivity"]
    conditional_instance = summary[
        "instance_connectivity_conditional_on_completed"
    ]
    type_match = summary["type_connectivity"]
    conditional_type = summary["type_connectivity_conditional_on_completed"]
    lines.extend(
        [
            "",
            (
                f"End-to-end exact MOFs: {exact['count']}/{exact['denominator']} "
                f"({exact['fraction']:.2%})."
            ),
            (
                "Conditional on composition and redecomposition completing: "
                f"{conditional_exact['count']}/{conditional_exact['denominator']} "
                "MOFs were exact."
            ),
            (
                f"End-to-end instance connectivity: {instance['matched_count']}/"
                f"{instance['reference_count']} expected BBs matched; "
                f"precision={instance['precision']:.4f}, "
                f"recall={instance['recall']:.4f}, F1={instance['f1']:.4f}."
            ),
            (
                "Completed-case instance connectivity: "
                f"{conditional_instance['matched_count']}/"
                f"{conditional_instance['reference_count']} expected BBs matched."
            ),
            (
                f"End-to-end type connectivity: {type_match['matched_count']}/"
                f"{type_match['reference_count']} input BB types matched; "
                "completed-case type connectivity: "
                f"{conditional_type['matched_count']}/"
                f"{conditional_type['reference_count']}."
            ),
            "",
            "Coordinates, atom ordering, rigid transforms, bond labels, and BB "
            "role names are ignored by the primary metric.",
            "",
        ]
    )
    failures = [record for record in records if record["status"] != "completed"]
    if failures:
        lines.extend(["## Failures", ""])
        for record in failures:
            lines.append(
                f"- {record['case']} [{record['failure_stage']}]: {record['error']}"
            )
        lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "decompositions",
        nargs="*",
        type=Path,
        help=(
            "Full node-edge decomposition directories. When omitted, use every "
            "directory under examples/node-edge-decomposed."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--accuracy", type=int, default=6)
    args = parser.parse_args()
    if args.accuracy < 1:
        parser.error("--accuracy must be positive")
    return args


def main() -> int:
    args = parse_args()
    sources = (
        [path.resolve() for path in args.decompositions]
        if args.decompositions
        else sorted(path.resolve() for path in DEFAULT_SOURCE.iterdir() if path.is_dir())
    )
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output}.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir(exist_ok=True)

    logger.disabled = True
    logging.getLogger("mofbb").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message="crystal system 'triclinic' is not interpreted.*",
    )
    warnings.filterwarnings(
        "ignore",
        message="Optimal rotation is not uniquely or poorly defined.*",
    )

    records = []
    for index, source in enumerate(sources, start=1):
        record = run_case(source, output, args.accuracy)
        records.append(record)
        match = record["instance_connectivity"] or {}
        print(
            f"[{index}/{len(sources)}] {record['case']}: "
            f"status={record['status']} "
            f"matched={match.get('matched_count', 0)}/"
            f"{record['expected_instance_count'] or 0} "
            f"exact={record['exact']}",
            flush=True,
        )

    summary = build_summary(records, args.accuracy)
    (output / "manifest.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(output, summary, records)
    exact = summary["exact_connectivity_roundtrips"]
    print(
        f"exact={exact['count']}/{exact['denominator']} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
