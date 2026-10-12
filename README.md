# MOFBB

Extract reusable building blocks from metal–organic frameworks (MOFs) and
assemble them into new MOFs with this Python toolkit.

## Install

Requires **Python 3.10+**. Run from the repository directory:

```bash
python -m pip install -e .
```

## Extract building blocks

```bash
mofbb-decompose structure.cif \
  --output-dir outputs/structure \
  --write-topology
```

Outputs include metal nodes, organic nodes, and linker edges as XYZ files:

- `bbs/`: one reusable representative per fragment group.
- `decomposition_summary.json`: fragment counts, groups, and file paths.
- `topology.cgd`: the periodic net, saved with `--write-topology`.

Cut bonds are capped with dummy `X` atoms for assembly. Solvent is kept by
default; add `--solvent-mode free` or `--solvent-mode all` to remove it.

## Build a new MOF

Choose a CGD topology and your building blocks. Replace these example filenames
and type IDs with your own:

```bash
mofbb-compose topology.cgd \
  --node-bb 0=node_type_0.xyz \
  --node-bb 1=node_type_1.xyz \
  --edge-bb 0,1=edge_type_0_1.xyz \
  --output outputs/new_mof.cif
```

Assign a block to every node type. CLI types start at `0` (`NODE 1` in CGD maps
to type `0`), and each node block needs one `X` site per connection. Edge blocks
are optional; omitted edges connect nodes directly.

Assembly checks connections, bond lengths, overlaps, and basic chemistry.
**3D periodic connectivity** is required by default; use
`--allow-low-dimensional` for layers, chains, or finite structures.

## Validate a CIF

```bash
python -m mofbb.compose.validate outputs/new_mof.cif
```

Checks 3D connectivity from coordinates. Add `--single-linker` to require one
organic linker type. Exit status is `0` on success and `1` on failure. Recheck
structures after external geometry optimization.

## Generate multiple MOFs

```bash
python scripts/generate_unique_mofs.py \
  --library /path/to/library \
  --output /path/to/new_run \
  --count 10
```

The library needs `building_blocks_manifest.csv` and a `topology/` directory.
The launcher excludes duplicates using the local `2m_mofs_v2` database and its
archives; use `--history /path/to/generation.sqlite3` for another history database.
Checks use assignment signatures and exact CIF hashes. Use a fresh output
folder when changing the library or validation policy.

## Help

```bash
mofbb-decompose --help
mofbb-compose --help
python -m mofbb.compose.validate --help
```

See [`examples/`](examples/) for sample building blocks and topologies.
