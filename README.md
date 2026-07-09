# MOFBB

MOFBB is a Python package for decomposing metal-organic framework (MOF) CIF files into MOFid SingleNode-inspired node and linker fragments capped with dummy `X` connection sites.

The package emits:

- capped node fragments as extended XYZ files;
- capped linker fragments as extended XYZ files;
- grouped building-block representatives;
- a P1 CGD topology file;
- a reconstruction map for reconnecting building-block slots; and
- a machine-readable decomposition summary.

> Status: research prototype.

## Installation

From a local checkout:

```bash
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e '.[dev]'
```

## Command-line usage

```bash
mofbb path/to/structure.cif --output-dir outputs/structure
```

Useful options:

```bash
mofbb path/to/structure.cif \
  --output-dir outputs/structure \
  --bond-scale 1.2 \
  --x-symbol X \
  --json
```

To keep the existing metal-node/linker decomposition, omit
`--decomposition` or pass `--decomposition single-node`. To split organic
linkers into smaller PORMAKE-style organic node and edge building blocks, use:

```bash
mofbb path/to/structure.cif \
  --decomposition node-edge \
  --output-dir outputs/structure-node-edge
```

## Python usage

```python
from mofbb import SingleNodePorDecomposer

result = SingleNodePorDecomposer("path/to/structure.cif").decompose()
summary = result.write("outputs/structure")
print(summary["num_nodes"], summary["num_linkers"])
```

```python
from mofbb import NodeEdgePorDecomposer

result = NodeEdgePorDecomposer("path/to/structure.cif").decompose()
summary = result.write("outputs/structure-node-edge")
print(
    summary["num_metal_nodes"],
    summary["num_organic_nodes"],
    summary["num_edges"],
)
```

In node-edge mode, the first-stage metal-node/linker boundary is preserved.
Each organic linker is then analyzed as a graph: ring/core blocks with more
than two connection ports become `organic_node` fragments, while paths with
exactly two connection ports become `edge` fragments. The summary records both
original metal-linker cut bonds and new organic internal cut bonds.


## Code layout

The implementation is split so each module has a narrow responsibility:

```text
src/mofbb/
├── decomposer.py      # Thin orchestration layer and public workflow class
├── edge_decomposer.py # Alternative metal-node, organic-node, edge workflow
├── assignment.py      # SingleNode-inspired atom assignment rules
├── bonding.py         # ASE neighbor-list bond graph and connectivity cleanup
├── fragments.py       # Connected components, fragment materialization, X caps
├── topology.py        # Collapsed topology vertices and edges
├── results.py         # Output summaries, CGD, groups, reconstruction map
├── geometry.py        # Periodic coordinate and unwrapping helpers
├── fingerprint.py     # Connectivity-aware fragment grouping hash
├── models.py          # Dataclasses used across the package
├── constants.py       # MOFid metal/nonmetal classification
└── type_aliases.py    # Shared typing aliases
```

The high-level `SingleNodePorDecomposer` class now reads as an execution recipe:
assign atoms, find cut bonds, build capped fragments, build topology records, and
return a `DecompositionResult`.

## Output layout

```text
outputs/structure/
├── all_fragments/
│   ├── node_001.xyz
│   └── linker_001.xyz
├── bbs/
│   ├── node_group_01.xyz
│   └── linker_group_01.xyz
├── linkers/
│   └── linker_001.xyz
├── nodes/
│   └── node_001.xyz
├── decomposition_summary.json
├── reconstruction_map.json
└── topology.cgd
```
