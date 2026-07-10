# MOFid-Style Solvent Removal Design

**Date:** 2026-07-10
**Status:** Approved design
**Target package:** MOFBB

## Goal

Add an opt-in, pure-Python solvent-removal workflow to MOFBB that follows the
topological solvent rules in MOFid while continuing to use MOFBB's existing
`SingleNodeAtomAssigner`. The workflow must distinguish free and bound solvent,
support `none`, `free`, and `all` modes, preserve periodic connection
directions, and write MOFid-named FSR/ASR artifacts.

## MOFid Reference Behavior

The behavioral reference is the C++ implementation under
`/nanolab/libo/lvebmof/mofid/src/`:

- `Deconstructor::SimplifyTopology()` in `deconstructor.cpp` repeatedly
  simplifies duplicate connections and processes zero- and one-connected
  pseudoatoms.
- A zero-connected pseudoatom is deleted as `free solvent`, regardless of its
  role.
- A one-connected `linker` pseudoatom is deleted as `bound solvent`.
- A one-connected `node` pseudoatom is merged into its neighbor when that
  neighbor is not also one-connected. A one-connected node attached to a
  one-connected linker is left in place for that pass.
- The loop repeats until no topology changes occur.
- `Deconstructor::WriteCIFs()` writes `free_solvent.cif`,
  `bound_solvent.cif`, `mof_fsr.cif`, and `mof_asr.cif`.
- FSR removes only free solvent. ASR removes free and bound solvent.

This feature ports those solvent topology semantics. It does not port the
entire MOFid node/linker deconstructor or Open Babel object model.

## Agreed Scope

1. MOFBB's existing SingleNode assignment remains the source of initial node
   and linker membership.
2. `solvent_mode="none"` preserves current behavior, including the existing
   `cleanup` option and largest-connected-component cleanup.
3. `solvent_mode="free"` and `solvent_mode="all"` bypass largest-component
   cleanup and analyze the complete bonded structure.
4. Both solvent classes are always detected when a solvent mode is enabled,
   even though only the selected cleaned structure proceeds to decomposition.
5. The implementation is pure Python and uses the package's existing ASE,
   NumPy, and NetworkX dependencies.
6. MOFid compatibility means equivalent artifact names, cell and atom
   membership, and FSR/ASR semantics. Byte-identical CIF formatting and Open
   Babel-specific bond-loop formatting are not required.

## Public API

Add the following constructor argument to both public decomposers:

```python
solvent_mode: Literal["none", "free", "all"] = "none"
```

This applies to:

- `SingleNodePorDecomposer`
- `NodeEdgePorDecomposer`
- `decompose_mof_with_edges`

The CLI gains:

```text
--solvent-mode {none,free,all}
```

`MofidSolventRemover` and `SolventRemovalResult` are public lazy exports from
`mofbb`. The standalone classifier interface is:

```python
analysis = MofidSolventRemover(atoms, bonds).classify()
clean_atoms, clean_bonds, selected_to_original = analysis.subset("all")
```

`SolventRemovalResult` contains:

- free-solvent components as tuples of original atom indices;
- bound-solvent components as tuples of original atom indices;
- FSR and ASR original atom-index tuples;
- access to the original `ase.Atoms` and `BondNetwork` needed for writing;
- a `subset(mode)` method for `free` and `all` that returns a reindexed bonded
  structure and a cleaned-index-to-original-index tuple.

The decomposition result types gain optional solvent metadata and the selected
index mapping. No solvent files or solvent JSON block are added in `none` mode,
so the default output layout remains backward compatible.

## Components and Responsibilities

### `src/mofbb/solvent.py`

Owns the solvent feature:

- public `SolventMode`, `MofidSolventRemover`, and `SolventRemovalResult`;
- internal topology-site and topology-connection records;
- component collapse and original-atom ownership;
- periodic connection normalization;
- MOFid-style fixed-point simplification;
- FSR/ASR atom-set construction and invariant validation;
- solvent metadata and CIF-writing helpers shared by both result types.

### `src/mofbb/bonding.py`

Adds one reusable bonded-subset helper. Given ordered original indices, it
returns the subset `ase.Atoms`, a reindexed `BondNetwork`, and the
cleaned-to-original mapping while preserving minimum-image vectors and source
bond types. Existing largest-component cleanup may call this helper to avoid
duplicating reindexing logic.

### Decomposers

`decomposer.py` and `edge_decomposer.py` coordinate preparation:

- In `none` mode, call `read_bonded_structure` exactly as today.
- In `free` or `all` mode, call it with `cleanup=False`, run the solvent
  classifier, select FSR or ASR, and decompose the selected bonded structure.
- Run SingleNode assignment again on the selected structure for the actual
  fragment decomposition.
- Attach the solvent result, mode, and index mapping to the returned result.

### Results and CLI

`results.py` and the node-edge result writer add the shared solvent JSON block
and four CIF outputs when solvent analysis is enabled. `cli.py` validates and
forwards `--solvent-mode`.

## Topology Construction

### Initial sites

1. Run `SingleNodeAtomAssigner` on the complete bonded structure.
2. Let linker atoms be the complement of node atoms.
3. Find connected components induced separately by the node and linker sets.
4. Collapse each component to a topology site with a stable identifier, role,
   and owned set of original atom indices.
5. Every original atom must belong to exactly one initial site.

Stable identifiers are assigned by role and minimum original atom index. They
provide deterministic processing independent of Python set ordering.

### Component image offsets

Periodic connection comparison must be performed after accounting for how
each collapsed component unwraps across the unit cell. For every site, traverse
its internal bond graph from a stable root and assign each atom an integer cell
image offset. The stored minimum-image bond vectors determine the offset
between adjacent atoms.

For a crossing bond from atom `i` in site A to atom `j` in site B:

1. Derive the bond's integer lattice shift from the difference between
   `BondNetwork.vector(i, j)` and the direct Cartesian displacement.
2. Let `offset_i` and `offset_j` be the component-unwrapping image offsets.
3. Define the collapsed-site translation from A to B as:

   ```text
   bond_shift + offset_i - offset_j
   ```

4. Normalize orientation by stable site identifier; reverse orientation also
   negates the translation.

Translations must be integral within numerical tolerance. A periodic structure
with a singular or unusable cell raises `ValueError` rather than silently
discarding periodic information.

### MOFid `SimplifyAxB` analogue

Connections are keyed by normalized endpoint sites plus normalized integer
translation:

- Connections with the same endpoints and translation are redundant
  nonperiodic duplicates and collapse to one topology connection.
- Connections between the same sites but through different translations are
  retained as distinct periodic connections.

Site degree is the number of retained topology connections, not merely the
number of distinct neighboring site identifiers.

The current `BondNetwork` stores one minimum-image vector per source atom pair.
The solvent layer preserves all distinct periodic connections represented by
different source atom pairs. It cannot recreate multiple image bonds that were
already collapsed into one source atom pair by the input bond layer; this is an
existing representation boundary, not a new solvent heuristic.

## Fixed-Point Solvent Classification

Process active sites in stable identifier order. Degree is recalculated from
the current topology immediately before each site is processed.

For each active site:

1. If its degree is one and its role is `node`:
   - leave it unchanged when its neighbor also has degree one;
   - otherwise merge its owned original atoms into the neighbor, retain the
     neighbor's role, and delete their connecting edge and the node site.
2. Else, if its degree is one and its role is `linker`, record its complete
   current atom ownership as one bound-solvent component and delete the site and
   its connection.
3. Else, if its degree is zero, record its complete current atom ownership as
   one free-solvent component and delete the site.

Repeat complete passes until no site is merged or deleted. This order mirrors
the C++ branch order and permits one removal to expose another terminal or
isolated component on a later pass. Atoms merged from a node into a linker are
classified with that linker if it is subsequently removed. Within each solvent
class, emitted components are ordered by their minimum original atom index.

## Atom Products and Invariants

Let `F` be all original free-solvent indices and `B` all original bound-solvent
indices:

```text
FSR = all original indices - F
ASR = all original indices - F - B
```

All output index tuples retain original CIF atom order. The classifier validates:

- every original atom was owned by exactly one initial site;
- free components are mutually disjoint;
- bound components are mutually disjoint;
- free and bound atoms are disjoint;
- `FSR union F` equals the original atom set;
- `ASR union F union B` equals the original atom set;
- subset mappings are bijective and ordered;
- every retained bond has both endpoints retained and is reindexed once.

If the selected mode removes every atom, decomposition raises a descriptive
`ValueError` containing the free and bound atom counts.

## Output Contract

When solvent analysis is enabled, `result.write(output_dir)` writes:

- `free_solvent.cif`: all original atoms classified as free solvent;
- `bound_solvent.cif`: all original atoms classified as bound solvent;
- `mof_fsr.cif`: the original structure minus free solvent;
- `mof_asr.cif`: the original structure minus free and bound solvent.

All files retain the original cell, periodic flags, coordinates, and original
atom order. Empty solvent selections still produce valid, readable empty CIFs.

The decomposition summary gains:

```json
{
  "solvent": {
    "mode": "all",
    "free": {
      "num_components": 1,
      "num_atoms": 3,
      "components": [[120, 121, 122]],
      "path": "free_solvent.cif"
    },
    "bound": {
      "num_components": 1,
      "num_atoms": 6,
      "components": [[114, 115, 116, 117, 118, 119]],
      "path": "bound_solvent.cif"
    },
    "fsr": {"num_atoms": 120, "path": "mof_fsr.cif"},
    "asr": {"num_atoms": 114, "path": "mof_asr.cif"},
    "selected_to_original": [0, 1, 2]
  }
}
```

The example values are illustrative; the schema and field meanings are fixed.
Existing fragment `source_indices` remain indices into the selected/reindexed
structure. `selected_to_original[fragment_source_index]` maps them back to the
input CIF.

## Error Handling

- Reject an unsupported solvent mode with `ValueError` at construction time.
- Reject inconsistent atom ownership, overlapping solvent classes, invalid
  subset mappings, or unmapped connection endpoints with descriptive
  `ValueError`s.
- Reject periodic translation calculations when the cell is singular or a
  derived shift is not integral within tolerance.
- Reject FSR/ASR selection that removes the entire structure before atom-role
  assignment or fragment construction.
- Structures with no detected solvent are valid: both solvent files are empty,
  and FSR and ASR contain the original structure.

## Testing Strategy

All production behavior is implemented test-first.

### Unit topology tests

Use compact synthetic `ase.Atoms` and `BondNetwork` fixtures to prove:

- a zero-connected component becomes free solvent;
- a one-connected linker becomes bound solvent;
- a two-connected linker remains framework;
- an eligible one-connected node merges into its neighbor;
- a one-connected node attached to a one-connected linker is not prematurely
  merged;
- removal exposes new bound and free sites over multiple passes;
- duplicate nonperiodic connections collapse;
- connections through different periodic images remain distinct;
- node atoms merged into a later-deleted linker inherit the linker's solvent
  classification.

### Atom and bond accounting tests

Prove:

- free and bound classes are disjoint;
- FSR and ASR conservation identities hold;
- component membership is stable and deterministic;
- bonded subsetting preserves source bond types and oriented minimum-image
  vectors;
- cleaned-to-original mappings are ordered and bijective;
- invalid modes, unusable periodic cells, and all-atoms-removed cases fail with
  the specified messages.

### Integration tests

Cover:

- `none`, `free`, and `all` in `SingleNodePorDecomposer`;
- the same preparation path in `NodeEdgePorDecomposer`;
- CLI parsing and forwarding;
- summary JSON schema and values;
- creation of all four CIF files, including empty solvent files;
- reading each emitted CIF back with ASE and comparing atom count and formula;
- default-mode regression behavior and the complete existing test suite.

### MOFid reference audit

Run the Python classifier against compact topology scenarios derived directly
from `Deconstructor::SimplifyTopology()`. Also audit bundled MOFid CIF/output
pairs where practical. Full-CIF differences caused by the explicitly retained
MOFBB SingleNode assignment are reported as assignment differences rather than
silently changing solvent rules.

## Documentation

Update the README with:

- the three modes and their FSR/ASR meanings;
- Python and CLI examples;
- output files and summary fields;
- the fact that MOFid treats terminal/capping linkers as bound solvent;
- the distinction between `cleanup` and topology-based solvent removal;
- the behavior-level scope relative to the original MOFid implementation.

## Completion Criteria

The feature is complete only when:

1. Both decomposers and the CLI support all three modes.
2. Default `none` behavior remains backward compatible.
3. Free and bound solvent follow the approved iterative topology rules.
4. Distinct periodic connection translations affect degree correctly.
5. FSR/ASR atom conservation and cleaned-index mapping are verified.
6. All four MOFid-named CIF artifacts and JSON metadata are emitted for enabled
   solvent modes.
7. Focused tests and the entire existing test suite pass without warnings or
   failures attributable to the feature.
8. README documentation describes the implemented behavior and limitations.
