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

### Periodic rod components

Infinite one-dimensional metal components are exported as finite
crystallographic repeats. Bonds that close the rod through a neighboring cell
become translated topology self-edges, with their exact cell shifts recorded in
`reconstruction_map.json`. This representation preserves the input cell and
bond multiplicity; it does not try to infer a smaller chemistry-specific SBU.

The output distinguishes two coordination views:

- `physical_cn` and the X atoms in each slot's `exact_instance_path` XYZ file
  describe every cut bond and should be used for exact reconstruction.
- `logical_cn`, `logical_kind`, and `logical_ports` quotient parallel bonds
  that connect the same neighboring repeat. They are the better view for
  interpreting topology or matching reusable building-block roles.

For example, an intact linker may contain six physical X sites but have two
logical ports containing three X sites each. Its exact reconstruction still
uses all six X atoms, while its logical role is a 2-connected edge. Exact
per-instance X membership is stored on reconstruction slots; grouped building
block metadata stores only invariant logical-port sizes.

The files under `bbs/` are fingerprint-group representatives for inspection or
library matching. Equivalent instances can have different local atom/X index
orders, and schema 2 does not emit a representative-to-instance permutation.
Consequently, a grouped representative is not an exact drop-in replacement for
`slots[].exact_instance_path`; this limitation is recorded by the
`geometry_contract` object in `reconstruction_map.json`.

`topology.cgd` exposes the translated geometry, but a stock PORMAKE reader may
not preserve every periodic self-edge or parallel edge. Treat
`reconstruction_map.json` schema 2 as the authoritative reconstruction record
for periodic rods.


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
