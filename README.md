# MOFBB

MOFBB is a Python toolkit for decomposing periodic metal-organic frameworks
(MOFs) into reusable, `X`-capped building blocks and composing those blocks
back into periodic structures.

It supports two complementary workflows:

- **Decomposition:** split an input CIF into metal-node, organic-node, and edge
  fragments, group equivalent fragments, and export topology and
  reconstruction metadata.
- **Composition:** either reconstruct the original per-instance geometry from
  a decomposition directory or assemble selected building blocks on a CGD
  topology.

> **Status:** research prototype. Inspect generated structures and validation
> metadata before using them in downstream calculations.

## Features

- Reads periodic CIF structures through ASE.
- Uses MOFid-style metal classification and atom-assignment rules.
- Optionally removes MOFid-style free or all solvent before decomposition.
- Caps every cut bond with an `X` connection site.
- Decomposes MOFs into metal nodes, organic nodes, and edges.
- Opens infinite one-dimensional metal components into finite
  crystallographic repeats while retaining their cell translations.
- Writes extended XYZ building blocks, a symmetry-preserving CGD topology,
  JSON summaries, and a reconstruction map.
- Reconstructs decomposition outputs without requiring JAX.
- Performs symmetry-constrained optimization of topological-node positions and
  lattice parameters, building-block placement, and geometric validation.

## Installation

MOFBB requires Python 3.10 or newer. From a local checkout, install the package
in editable mode:

```bash
python -m pip install -e .
```

Install the test and lint dependencies for development:

```bash
python -m pip install -e '.[dev]'
```

The installed commands are `mofbb-decompose` for decomposition and
`mofbb-compose` for composition.

## Quick start

Decompose a CIF using the default node-edge workflow:

```bash
mofbb-decompose path/to/structure.cif --output-dir outputs/structure
```

Print command-specific options with:

```bash
mofbb-decompose --help
mofbb-compose --help
```

The same interfaces are available as Python modules:

```bash
python -m mofbb.decompose --help
python -m mofbb.compose --help
```

## Decomposition

### Node-edge workflow

Node-edge decomposition analyzes each organic linker as a graph.
Organic cores with more than two connection sites become `organic_node`
fragments, while paths with exactly two connection sites become `edge`
fragments:

```bash
mofbb-decompose structure.cif \
  --output-dir outputs/node-edge
```

Common options include:

| Option | Default | Meaning |
| --- | --- | --- |
| `--bond-scale FLOAT` | `1.2` | Multiplier applied to ASE natural covalent cutoffs |
| `--x-symbol SYMBOL` | `X` | Dummy atom used to cap cut bonds |
| `--solvent-mode MODE` | `none` | Keep solvent, remove `free`, or remove `all` solvent |
| `--no-cleanup` | off | Keep all connected components |
| `--write-topology` | off | Emit the symmetry-aware `topology.cgd` file |
| `--json` | off | Print the reduced CLI output summary to standard output |

Keep the default `X` symbol when the output will be passed to
`mofbb-compose` as a topology assignment, whose connection-point handling is
based on `X` atoms.

## Composition

MOFBB exposes two distinct composition paths.

### Exact reconstruction

Pass a full Python-API decomposition output directory (or one of the prepared
example directories) to restore its per-instance fragments, remove the `X`
caps, and reconnect the cut bonds described by `reconstruction_map.json`:

```bash
mofbb-compose examples/singlenode-decomposed/UiO-67 \
  --output outputs/reconstructed.cif
```

The reduced `mofbb-decompose` directory intentionally omits per-instance
fragments and `reconstruction_map.json`, so it is intended for reusable BB
collection and topology-driven assembly rather than exact reconstruction.

Building-block assignments cannot be supplied in this mode. Exact
reconstruction uses each slot's `exact_instance_path`, not the grouped
representative under `bbs/`.

### Topology-driven assembly

Use `--write-topology` during decomposition when a generated CGD will be used
for topology-driven assembly.

Pass a CGD file and assign a building block to every topology node type:

```bash
mofbb-compose topology.cgd \
  --node-bb 0=node_type_0.xyz \
  --node-bb 1=node_type_1.xyz \
  --edge-bb 0,1=edge_type_0_1.xyz \
  --accuracy 6 \
  --output outputs/composed.cif
```

Assignments use `TYPE=PATH` for nodes and `LEFT_TYPE,RIGHT_TYPE=PATH` for
edges. Edge assignments are optional; an omitted edge type produces a direct
node-to-node connection. Each node building block must have the same number of
`X` connection points as the coordination number of its topology slot.
The first field after `NODE` is a one-based type ID: repeated IDs use the same
node assignment (internally, `NODE 1` is type `0`).

The builder removes paired `X` atoms and fuses their real-atom anchors. It
rejects an assembly when any of these checks fails:

- a new fused bond directly joins carbon to a metal;
- paired `X` atoms are more than `0.3 Å` apart under the minimum-image
  convention;
- a fused bond is longer than `1.2` times the sum of the ASE covalent radii; or
- two real atoms are closer than their smaller ASE covalent radius under the
  minimum-image convention.

These thresholds can be adjusted with `--x-mid-tolerance` and
`--fused-bond-scale`, and `--atomic-overlap-scale`; setting the last option to
zero disables only the atomic-overlap gate. The overlap gate checks all real
atom pairs, including pairs within one building block. Use `--no-wrap` to keep
the composed coordinates outside the primary unit cell.

## Python API

### Decompose a structure

```python
from mofbb.decompose import NodeEdgeDecomposer

result = NodeEdgeDecomposer(
    "path/to/structure.cif",
    solvent_mode="all",
).decompose()
summary = result.write(
    "outputs/node-edge",
    write_topology=True,
)

print(
    summary["num_metal_nodes"],
    summary["num_organic_nodes"],
    summary["num_edges"],
)
```

### Reconstruct a decomposition

```python
from mofbb.compose import Reconstructor

framework = Reconstructor("outputs/structure").reconstruct()
framework.write_cif("outputs/reconstructed.cif")
```

### Build on a topology

```python
from mofbb.compose import Builder, BuildingBlock, Topology

topology = Topology("topology.cgd")
node_blocks = {
    0: BuildingBlock("node_type_0.xyz"),
    1: BuildingBlock("node_type_1.xyz"),
}
edge_blocks = {
    (0, 1): BuildingBlock("edge_type_0_1.xyz"),
}

framework = Builder().build_by_type(topology, node_blocks, edge_blocks)
framework.write_cif("outputs/composed.cif")
```

## Output files

The CLI and Python API write the same reconstruction-oriented layout; the CLI
also copies the source CIF into the output directory:

```text
outputs/structure/
├── bbs/                        # One representative per fingerprint group
├── edges/                      # Exact per-instance edge geometries
├── metal_nodes/                # Exact per-instance metal nodes
├── organic_nodes/              # Exact per-instance organic nodes
├── decomposition_summary.json # Counts, groups, and paths
├── reconstruction_map.json    # Slot and cut-bond reconstruction data
└── topology.cgd               # Optional; enable with --write-topology
```

Every fragment XYZ stores its connection-point indices and bond records in the
extended XYZ comment and bond sections.

The main metadata files serve different purposes:

- `decomposition_summary.json` is the user-facing inventory of groups and,
  when requested, topology metadata.
- `reconstruction_map.json` records exact slot instances, source atom indices,
  paired `X` sites, and cut-bond translations.
- Optional `topology.cgd` records the collapsed periodic net with its validated
  graph symmetry and explicit symmetry operations.

These files intentionally use two different fragment centers. A CGD `NODE`
position is the centroid of the fragment's connection (`X`) sites. Each
reconstruction slot instead stores the wrapped real-atom centroid in
`exact_frac_position`.

## Periodic components and geometry contract

Infinite one-dimensional metal components are exported as finite
crystallographic repeats. Bonds that close a repeat through a neighboring cell
become translated topology self-edges, with their cell shifts stored in
`reconstruction_map.json`.

The metadata distinguishes two coordination views:

- `physical_cn` and the `X` atoms in `exact_instance_path` describe every cut
  bond required for exact reconstruction.
- `logical_cn`, `logical_kind`, and `logical_ports` merge parallel physical
  bonds that connect the same neighboring repeat and are more useful for
  interpreting reusable building-block roles.

For example, a fragment can have six physical `X` sites grouped into two
logical ports of size three. Its exact reconstruction uses all six sites, while
its logical role is two-connected.

Files under `bbs/` are fingerprint-group representatives intended for
inspection and library matching. Equivalent instances can use different local
atom or `X` index orderings, and schema 2 does not provide a
representative-to-instance permutation. Therefore, a grouped representative is
not an exact replacement for a slot's `exact_instance_path`.

For periodic rods and structures with self-edges or parallel edges, treat
`reconstruction_map.json` as authoritative. A generic CGD/PORMAKE reader may
not preserve all translated edge multiplicities.

## Examples

Prepared legacy decomposition artifacts for CPO-27-Ni, HKUST-1, MOF-5,
UiO-67, and Zn2-TTFB are available under [`examples/`](examples/).
They can be used to inspect the output schema or test reconstruction directly:

```bash
mofbb-compose examples/singlenode-decomposed/UiO-67 \
  --output outputs/UiO-67-reconstructed.cif
```

## Development

The source tree is organized by workflow:

```text
src/mofbb/
├── cli.py                   
├── compose/
│   ├── cli.py               # Composition CLI
│   ├── reconstructor.py     # Exact reconstruction
│   ├── builder.py           # Topology-driven assembly
│   ├── building_block.py    # Building-block model
│   ├── topology.py          # CGD topology model
│   ├── node_lattice.py      # Node-position and lattice optimization
│   ├── geometry_parameterization.py # Symmetry-constrained coordinates
│   └── placement/fixed_cell.py      # Fixed-cell framework fitting
└── decompose/
    ├── cli.py               # Decomposition CLI
    ├── decomposer.py        # Node-edge workflow
    ├── linker.py            # Organic node/edge splitting
    ├── ports.py             # Logical connection-port assignment
    ├── bonding.py           # Bond graph construction and cleanup
    ├── fragments.py         # Fragment generation and X caps
    ├── pbc.py               # Periodic cuts and repeat opening
    └── results.py           # Output and reconstruction metadata
```
