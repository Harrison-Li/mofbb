# MOFBB

MOFBB is a Python toolkit for decomposing periodic metal-organic frameworks
(MOFs) into reusable, `X`-capped building blocks and composing those blocks
back into periodic structures.

It supports two complementary workflows:

- **Decomposition:** split an input CIF into node/linker or
  metal-node/organic-node/edge fragments, group equivalent fragments, and
  export the corresponding topology and reconstruction metadata.
- **Composition:** either reconstruct the original per-instance geometry from
  a decomposition directory or assemble selected building blocks on a CGD
  topology.

> **Status:** research prototype. Inspect generated structures and validation
> metadata before using them in downstream calculations.

## Features

- Reads periodic CIF structures through ASE.
- Uses MOFid-style metal classification and atom-assignment rules.
- Caps every cut bond with an `X` connection site.
- Supports single-node and node-edge decomposition.
- Optionally removes free or all detected solvent before decomposition.
- Opens infinite one-dimensional metal components into finite
  crystallographic repeats while retaining their cell translations.
- Writes extended XYZ building blocks, a P1 CGD topology, JSON summaries, and
  a schema-v2 reconstruction map.
- Reconstructs decomposition outputs without requiring JAX.
- Performs topology scaling, building-block orientation, and geometric quality
  checks for topology-driven assembly.

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

The installed commands are `mofbb` for decomposition and `mofbb-compose` for
composition.

## Quick start

Decompose a CIF using the default single-node workflow:

```bash
mofbb path/to/structure.cif --output-dir outputs/structure
```

Reconstruct a CIF from the generated directory:

```bash
mofbb-compose outputs/structure --output outputs/reconstructed.cif
```

Print command-specific options with:

```bash
mofbb --help
mofbb-compose --help
```

The same interfaces are available as Python modules:

```bash
python -m mofbb.decompose --help
python -m mofbb.compose --help
```

## Decomposition

### Workflows

| Mode | Command | Building-block roles |
| --- | --- | --- |
| Single-node (default) | `--decomposition single-node` | Metal-containing `node` fragments and organic `linker` fragments |
| Node-edge | `--decomposition node-edge` | `metal_node`, `organic_node`, and two-connected `edge` fragments |

Single-node decomposition preserves each organic linker as one fragment:

```bash
mofbb structure.cif \
  --decomposition single-node \
  --output-dir outputs/single-node
```

Node-edge decomposition further analyzes each organic linker as a graph.
Organic cores with more than two connection sites become `organic_node`
fragments, while paths with exactly two connection sites become `edge`
fragments:

```bash
mofbb structure.cif \
  --decomposition node-edge \
  --output-dir outputs/node-edge
```

Common options include:

| Option | Default | Meaning |
| --- | --- | --- |
| `--bond-scale FLOAT` | `1.2` | Multiplier applied to ASE natural covalent cutoffs |
| `--x-symbol SYMBOL` | `X` | Dummy atom used to cap cut bonds |
| `--solvent-mode {none,free,all}` | `none` | Solvent treatment before decomposition |
| `--no-cleanup` | off | Keep all connected components in legacy `none` solvent mode |
| `--json` | off | Print the full decomposition summary to standard output |

Keep the default `X` symbol when the output will be passed to
`mofbb-compose`, whose connection-point handling is based on `X` atoms.

### Solvent handling

MOFBB provides three solvent modes:

| Mode | Structure decomposed |
| --- | --- |
| `none` | No explicit solvent classification; by default, retain the largest connected component |
| `free` | Free-solvent-removed (FSR) structure; bound solvent remains |
| `all` | All-solvent-removed (ASR) structure; both free and bound solvent are removed |

For example:

```bash
mofbb structure.cif \
  --solvent-mode all \
  --output-dir outputs/desolvated
```

When `free` or `all` is selected, the output also contains
`free_solvent.cif`, `bound_solvent.cif`, `mof_fsr.cif`, and `mof_asr.cif`, and
the atom mapping back to the original input is recorded in
`decomposition_summary.json`.

## Composition

MOFBB exposes two distinct composition paths.

### Exact reconstruction

Pass a decomposition output directory to restore its per-instance fragments,
remove the `X` caps, and reconnect the cut bonds described by
`reconstruction_map.json`:

```bash
mofbb-compose outputs/structure \
  --output outputs/reconstructed.cif
```

Building-block assignments cannot be supplied in this mode. Exact
reconstruction uses each slot's `exact_instance_path`, not the grouped
representative under `bbs/`.

### Topology-driven assembly

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

The builder removes paired `X` atoms and fuses their real-atom anchors. It
rejects an assembly when either of its default geometry checks fails:

- paired `X` atoms are more than `0.3 Å` apart under the minimum-image
  convention; or
- a fused bond is longer than `1.2` times the sum of the ASE covalent radii.

These thresholds can be adjusted with `--x-mid-tolerance` and
`--fused-bond-scale`. Use `--no-wrap` to keep the composed coordinates outside
the primary unit cell.

## Python API

### Decompose a structure

```python
from mofbb.decompose import SingleNodeDecomposer

result = SingleNodeDecomposer(
    "path/to/structure.cif",
    bond_scale=1.2,
    solvent_mode="none",
).decompose()

summary = result.write("outputs/structure")
print(summary["num_nodes"], summary["num_linkers"])
```

Use `NodeEdgeDecomposer` for the node-edge workflow:

```python
from mofbb.decompose import NodeEdgeDecomposer

result = NodeEdgeDecomposer("path/to/structure.cif").decompose()
summary = result.write("outputs/node-edge")

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

A single-node decomposition has the following layout:

```text
outputs/structure/
├── all_fragments/              # Copies of every fragment
├── bbs/                        # One representative per fingerprint group
├── linkers/                    # Exact per-instance linker geometries
├── nodes/                      # Exact per-instance node geometries
├── decomposition_summary.json # Counts, groups, paths, and validation data
├── reconstruction_map.json    # Authoritative slot/edge reconstruction data
└── topology.cgd               # P1 topology and translated edge geometry
```

Node-edge output uses `metal_nodes/`, `organic_nodes/`, and `edges/`; it can
also contain `linkers/` for retained bound-solvent accessories. Every fragment
XYZ stores its connection-point indices and bond records in the extended XYZ
comment and bond sections.

The main metadata files serve different purposes:

- `decomposition_summary.json` is the user-facing inventory of fragments,
  groups, solvent selections, and topology counts.
- `reconstruction_map.json` records exact slot instances, source atom indices,
  paired `X` sites, cut-bond translations, and reconstruction validation.
- `topology.cgd` provides a P1 view of the collapsed periodic net.

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
not an exact replacement for a slot's `exact_instance_path`; this limitation is
recorded in the reconstruction map's `geometry_contract` object.

For periodic rods and structures with self-edges or parallel edges, treat
`reconstruction_map.json` as authoritative. A generic CGD/PORMAKE reader may
not preserve all translated edge multiplicities.

## Examples

Prepared single-node and node-edge decomposition artifacts for CPO-27-Ni,
HKUST-1, MOF-5, UiO-67, and Zn2-TTFB are available under [`examples/`](examples/).
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
│   └── scaler.py            # NumPy/SciPy topology scaling
└── decompose/
    ├── cli.py               # Decomposition CLI
    ├── decomposer.py        # Single-node workflow
    ├── edge_decomposer.py   # Node-edge workflow
    ├── bonding.py           # Bond graph construction and cleanup
    ├── fragments.py         # Fragment generation and X caps
    ├── periodicity.py       # Periodic cuts and repeat opening
    ├── results.py           # Output and reconstruction metadata
    └── solvent.py           # Solvent classification and export
```
