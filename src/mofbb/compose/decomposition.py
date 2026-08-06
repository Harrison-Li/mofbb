"""Load grouped building blocks from a decorated abstract topology."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from mofbb.compose.building_block import BuildingBlock
from mofbb.compose.topology import Topology


@dataclass
class AbstractTopologyInput:
    """Topology and grouped BB templates loaded without reconstruction data."""

    topology: Topology
    templates: dict[str, BuildingBlock]
    slot_bb_ids: tuple[str | None, ...]
    template_paths: dict[str, str]

    @classmethod
    def from_directory(cls, decomposition_dir: str | Path):
        """Read ``topology.cgd`` and grouped BBs from a decomposition directory."""

        root = Path(decomposition_dir)
        summary_path = root / "decomposition_summary.json"
        if not summary_path.is_file():
            raise ValueError(f"Decomposition directory {root} has no summary file.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        topology_record = summary.get("topology")
        if not topology_record:
            raise ValueError(
                f"Decomposition directory {root} has no abstract topology."
            )
        topology_path = root / topology_record.get("path", "topology.cgd")
        topology = Topology(topology_path)

        group_paths = {}
        for group in summary.get("groups", ()):
            bb_id = str(group["bb_id"])
            if bb_id in group_paths:
                raise ValueError(f"Duplicate grouped BB identifier {bb_id!r}.")
            group_paths[bb_id] = str(group["path"])

        node_bb_ids = topology.atoms.info.get("node_bb_ids", {})
        edge_bb_ids = topology.atoms.info.get("edge_bb_ids", {})
        node_slots = {int(index) for index in topology.node_indices}
        slot_bb_ids = []
        for slot, raw_type in enumerate(topology.atoms.get_tags()):
            bb_ids = node_bb_ids if slot in node_slots else edge_bb_ids
            bb_type = int(raw_type)
            if bb_type not in bb_ids:
                raise ValueError(
                    f"Abstract topology slot {slot} has no declared BB type."
                )
            bb_id = bb_ids[bb_type]
            if bb_id is not None and bb_id not in group_paths:
                raise ValueError(
                    f"Abstract topology slot {slot} references missing BB "
                    f"group {bb_id!r}."
                )
            slot_bb_ids.append(bb_id)

        referenced = sorted({bb_id for bb_id in slot_bb_ids if bb_id is not None})
        templates = {
            bb_id: BuildingBlock(root / group_paths[bb_id])
            for bb_id in referenced
        }
        for slot, bb_id in enumerate(slot_bb_ids):
            if bb_id is None:
                continue
            expected = int(topology.cn[slot])
            actual = templates[bb_id].n_connection_points
            if actual != expected:
                raise ValueError(
                    f"Topology slot {slot} requires {expected} connection points, "
                    f"but BB group {bb_id!r} has {actual}."
                )
        return cls(
            topology=topology,
            templates=templates,
            slot_bb_ids=tuple(slot_bb_ids),
            template_paths={bb_id: group_paths[bb_id] for bb_id in referenced},
        )

    def blocks(self) -> list[BuildingBlock | None]:
        """Return independent per-slot copies of the grouped templates."""

        return [
            None if bb_id is None else self.templates[bb_id].copy()
            for bb_id in self.slot_bb_ids
        ]
