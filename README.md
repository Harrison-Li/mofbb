# MOFBB

MOFBB decomposes metal-organic framework (MOF) CIF files into reusable
building blocks with dummy `X` connection sites and composes those outputs back
into periodic frameworks. It supports both single-node and metal-node,
organic-node, edge decomposition styles.

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

Both command-line modules can also be run without installed launchers:

```bash
python -m mofbb.decompose --help
python -m mofbb.compose --help
```

## Composition

Reconstruct the exact per-slot geometry from a decomposition output directory:

```bash
mofbb-compose outputs/structure --output reconstructed.cif
```

The composer also supports topology-driven assembly with explicitly assigned
building blocks:

```bash
mofbb-compose topology.cgd \
  --node-bb 0=node_type_0.xyz \
  --node-bb 1=node_type_1.xyz \
  --edge-bb 0,1=edge_type_0_1.xyz \
  --output composed.cif
```

JAX is not required; topology scaling uses NumPy and SciPy.

## Python usage

```python
from mofbb.decompose import SingleNodeDecomposer

result = SingleNodeDecomposer("path/to/structure.cif").decompose()
summary = result.write("outputs/structure")
print(summary["num_nodes"], summary["num_linkers"])
```

```python
from mofbb.decompose import NodeEdgeDecomposer

result = NodeEdgeDecomposer("path/to/structure.cif").decompose()
summary = result.write("outputs/structure-node-edge")
print(
    summary["num_metal_nodes"],
    summary["num_organic_nodes"],
    summary["num_edges"],
)
```

```python
from mofbb.compose import Reconstructor

framework = Reconstructor("outputs/structure").reconstruct()
framework.write_cif("reconstructed.cif")
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
├── __init__.py             # Top-level public API compatibility exports
├── cli.py                  # Compatibility decomposition entry point
├── compose/
│   ├── __init__.py         # Composition public API
│   ├── cli.py              # Composition command-line interface
│   ├── reconstructor.py    # Exact decomposition-output reconstruction
│   ├── builder.py          # Topology-driven framework assembly
│   ├── building_block.py   # Building-block geometry and connectivity
│   ├── topology.py         # CGD topology model
│   ├── neighbor.py         # Scale-independent topology connectivity
│   └── scaler.py           # NumPy/SciPy topology scaling
└── decompose/
    ├── __init__.py         # Decomposition public API
    ├── cli.py              # Decomposition command-line interface
    ├── decomposer.py       # SingleNode orchestration and workflow class
    ├── edge_decomposer.py  # Metal-node, organic-node, edge workflow
    ├── assignment.py       # SingleNode-inspired atom assignment rules
    ├── bonding.py          # Bond graph and connectivity cleanup
    ├── constants.py        # MOFid metal/nonmetal classification
    ├── fingerprint.py      # Connectivity-aware fragment grouping hash
    ├── fragments.py        # Components, fragment materialization, and X caps
    ├── geometry.py         # Periodic coordinate and unwrapping helpers
    ├── models.py           # Decomposition dataclasses
    ├── periodicity.py      # Periodic connectivity and repeat opening
    ├── results.py          # Summaries, CGD, groups, and reconstruction map
    ├── solvent.py          # MOFid-style solvent removal
    ├── topology.py         # Collapsed topology vertices and edges
    └── type_aliases.py     # Shared decomposition typing aliases
```

The high-level `SingleNodeDecomposer` class now reads as an execution recipe:
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
