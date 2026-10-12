"""Ordered metal-substitution mode for generate_unique_mofs.py.

This constructs hypothetical chemistries at fixed mofbb template geometry.
It does not optimize substituted structures or establish charge neutrality.
Topology folder grouping is deliberately a separate post-processing step.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import warnings

import igraph as ig
import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, chemical_symbols
from ase.io.cif import parse_cif
from ase.neighborlist import neighbor_list
from pymatgen.core import Element
from threadpoolctl import threadpool_limits
from mofbb.compose import Builder, BuildingBlock, Topology
from mofbb.compose.framework import Framework
from mofbb.compose.connectivity import validate_3d_connectivity
from mofbb.compose.linkers import validate_single_linker
from mofbb.compose.chemistry import nonmetal_neighbor_counts, validate_coordination
from single_linker_mof_generator import configure_quiet_builder_logging

VERSION = 1
_WORK = None

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def canonical_key(numbers, edges):
    graph = ig.Graph(n=len(numbers), edges=np.asarray(edges, dtype=int).reshape(-1, 2).tolist())
    graph.vs['number'] = list(map(int, numbers))
    canonical = graph.permute_vertices(graph.canonical_permutation(color=graph.vs['number']))
    key = (tuple(canonical.vs['number']), tuple(sorted(tuple(sorted(e)) for e in canonical.get_edgelist())))
    # Same encoding as mofbb.compose.linkers._molecule_key consumers.
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()

def read_cif(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        block = next(parse_cif(str(path)))
        atoms = block.get_atoms()
    labels = list(block['_atom_site_label'])
    assert len(labels) == len(atoms) and len(set(labels)) == len(labels)
    index = {label: i for i, label in enumerate(labels)}
    bonds = np.array([[index[a], index[b]] for a, b in zip(
        block['_geom_bond_atom_site_label_1'], block['_geom_bond_atom_site_label_2'], strict=True)], dtype=int)
    shifts = np.array([[0, 0, 0] if s == '.' else [int(v)-5 for v in s.split('_')[1]]
                       for s in block['_geom_bond_site_symmetry_2']], dtype=int)
    return atoms, bonds, shifts, labels

def node_model(bonds, indices):
    local = {int(v): i for i, v in enumerate(indices)}
    internal, anchors = [], []
    for a, b in bonds:
        if int(a) in local and int(b) in local:
            internal.append(tuple(sorted((local[int(a)], local[int(b)]))))
        elif int(a) in local:
            anchors.append(local[int(a)])
        elif int(b) in local:
            anchors.append(local[int(b)])
    return {'internal': sorted(internal), 'anchors': sorted(anchors)}

def node_key(numbers, model):
    edges = model['internal'] + [(anchor, len(numbers)+i) for i, anchor in enumerate(model['anchors'])]
    return canonical_key(list(numbers) + [0]*len(model['anchors']), edges)

def prepare_template(plan, output):
    assets = output / 'template'
    assets.mkdir(exist_ok=True)
    metal, organic = BuildingBlock(plan['template_metal_path']), BuildingBlock(plan['organic_path'])
    assert organic.n_connection_points == 3 and metal.n_connection_points == 4
    metal.validate_connectivity(); organic.validate_connectivity()
    topology = Topology(plan['topology_path'])
    frame = Builder().build_by_type(topology, {plan['metal_node_type']: metal, plan['organic_node_type']: organic}, accuracy=4)
    maps, offset = [], 0
    placed_topology = frame.info['topology']
    for slot, block in enumerate(frame.info['located_bbs']):
        if block is None:
            continue
        real = np.flatnonzero(block.atoms.numbers != 0)
        if slot in placed_topology.node_indices and int(placed_topology.node_types[slot]) == plan['metal_node_type']:
            assert np.array_equal(block.atoms.numbers, metal.atoms.numbers)
            maps.append(list(range(offset, offset+len(real))))
        offset += len(real)
    assert offset == len(frame.atoms) and maps
    # Avoid printing inherited chemical-symmetry or optimization claims in derived CIFs.
    clean = Framework(frame.atoms, frame.bonds, frame.bond_types,
                      {'require_3d': True, 'scale': 1.2}, wrap=False, bond_shifts=frame.bond_shifts)
    clean.write_cif(assets/'parent.cif')
    atoms, bonds, shifts, labels = read_cif(assets/'parent.cif')
    assert np.array_equal(atoms.numbers, frame.atoms.numbers)
    models = [node_model(bonds, indices) for indices in maps]
    assert all(model == models[0] for model in models)
    assert len(models[0]['anchors']) == 4
    parent_key = canonical_key(metal.atoms.numbers, metal.bonds)
    assert node_key(atoms.numbers[maps[0]], models[0]) == parent_key
    real = organic.atoms.numbers != 0
    indices = np.flatnonzero(real); lookup = {v: i for i, v in enumerate(indices)}
    ligand_key = canonical_key(organic.atoms.numbers[real],
                              [(lookup[a], lookup[b]) for a, b in organic.bonds if real[a] and real[b]])
    q = validate_single_linker(atoms)
    assert q['linker_types'][0]['graph_sha256'] == ligand_key
    assert q['linker_types'][0]['formula'] == 'C18H9O6'
    q.update(validate_3d_connectivity(atoms, minimum_distance_cutoff=3.0))
    np.savez(assets/'parent_arrays.npz', numbers=atoms.numbers, positions=atoms.positions,
             cell=atoms.cell.array, bonds=bonds, shifts=shifts, metal_maps=np.array(maps))
    meta = {'node_model': models[0], 'labels': labels, 'metal_real_indices': np.flatnonzero(metal.atoms.numbers != 0).tolist(),
            'ligand_graph_sha256': ligand_key, 'template_quality': q, 'metal_copies_per_cell': len(maps),
            'parent_placement_max_rmsd': float(frame.info['max_rmsd'])}
    (assets/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    return assets

def init_worker(plan, output):
    global _WORK
    threadpool_limits(limits=1)
    warnings.filterwarnings('ignore')
    assets = Path(output)/'template'
    data = dict(np.load(assets/'parent_arrays.npz'))
    meta = json.loads((assets/'metadata.json').read_text())
    base = Atoms(numbers=data['numbers'], positions=data['positions'], cell=data['cell'], pbc=True)
    maximum = base.copy()
    maximum.numbers[maximum.numbers == 42] = 74
    maximum.numbers[maximum.numbers == 30] = 48
    assert np.all(covalent_radii[maximum.numbers] >= covalent_radii[base.numbers])
    i, j, d, shifts = neighbor_list('ijdS', maximum, np.maximum(1.2*covalent_radii[maximum.numbers], 1.5))
    metals = np.array([Element(s).is_metal for s in base.symbols])
    keep = (i <= j) & ~(metals[i] & metals[j])
    data.update(i=i[keep], j=j[keep], distances=d[keep], inferred_shifts=shifts[keep], all_i=i, all_j=j, all_distances=d)
    _WORK = (plan, Path(output), data, meta, (assets/'parent.cif').read_text().splitlines())

def generate_one(variant):
    plan, output, data, meta, lines = _WORK
    target = output/'.staging'/f"{variant['name']}.cif"
    try:
        assert sha(variant['path']) == variant['sha256']
        node = BuildingBlock(variant['path']); node.validate_connectivity()
        assert canonical_key(node.atoms.numbers, node.bonds) == variant['graph_sha256']
        node_numbers = node.atoms.numbers[meta['metal_real_indices']]
        numbers = data['numbers'].copy()
        for indices in data['metal_maps']:
            numbers[indices] = node_numbers
        new_labels = [f'{chemical_symbols[n]}{i}' for i, n in enumerate(numbers)]
        label_map = dict(zip(meta['labels'], new_labels, strict=True))
        output_lines = [
            '# Hypothetical ordered Mo/W and Zn/Cd substitution on a mofbb ctn template.',
            '# Fixed template geometry; no substituted-structure energy relaxation or charge assignment.',
            f"# Metal node graph SHA256: {variant['graph_sha256']}",
            f"# Metal substitution mask: {variant['mask']}",
            f"# Organic ligand: C18H9O6, three connection sites; SHA256 {plan['organic_sha256']}",
        ]
        for line in lines:
            fields = line.split()
            if line.startswith('data_'):
                line = f"data_{variant['name']}"
            elif fields and fields[0] in label_map:
                if len(fields) == 6:
                    index = meta['labels'].index(fields[0])
                    fields[0], fields[1] = new_labels[index], chemical_symbols[numbers[index]]
                elif len(fields) == 5 and fields[1] in label_map:
                    fields[0], fields[1] = label_map[fields[0]], label_map[fields[1]]
                else:
                    raise ValueError('Unrecognized template CIF record')
                line = ' '.join(fields)
            output_lines.append(line)
        target.write_text('\n'.join(output_lines)+'\n')
        atoms, bonds, shifts, labels = read_cif(target)
        assert np.array_equal(atoms.numbers, numbers)
        assert np.array_equal(atoms.positions, data['positions']) and np.array_equal(atoms.cell.array, data['cell'])
        assert np.array_equal(bonds, data['bonds']) and np.array_equal(shifts, data['shifts'])
        assert labels == new_labels and not np.any(numbers == 0)
        for indices in data['metal_maps']:
            assert np.array_equal(atoms.numbers[indices], node_numbers)
        actual_node_key = node_key(atoms.numbers[data['metal_maps'][0]], meta['node_model'])
        assert actual_node_key == variant['graph_sha256']
        # This neighbor-pair cache is geometric, not taken from intended CIF bonds.
        # Equal exported coordinates/cell are required above. The maximum-radii
        # search contains every candidate's distance-inferred bond.
        radii = covalent_radii[numbers]
        keep = data['distances'] < 1.2*(radii[data['i']] + radii[data['j']])
        inferred = np.column_stack((data['i'][keep], data['j'][keep]))
        images = data['inferred_shifts'][keep]
        q = validate_3d_connectivity(atoms, inferred, images)
        q.update(validate_single_linker(atoms, inferred, images))
        if q['num_linker_types'] != 1 or q['linker_types'][0]['graph_sha256'] != meta['ligand_graph_sha256']:
            raise ValueError('Exported ligand graph changed')
        if q['linker_types'][0]['formula'] != 'C18H9O6':
            raise ValueError('Exported ligand formula changed')
        validate_coordination(numbers, nonmetal_neighbor_counts(numbers, inferred), context='saved substituted CIF')
        minimum = float(data['all_distances'].min())
        if np.any(data['all_distances'] < np.minimum(radii[data['all_i']], radii[data['all_j']])):
            raise ValueError('Saved CIF has a covalent-radius overlap')
        q.update(num_atoms=len(atoms), formula=atoms.get_chemical_formula(mode='metal'),
                 minimum_atom_distance_angstrom=minimum, metal_node_graph_sha256=actual_node_key,
                 metal_copies_per_cell=len(data['metal_maps']), hypothetical=True,
                 energy_relaxed=False, charge_balance_validated=False,
                 geometry_source='fixed_mofbb_template', saved_cif_independently_checked=True,
                 neighbor_cache_basis='unchanged_exported_geometry_and_superset_covalent_cutoffs')
        return {'ok': True, 'variant': variant, 'path': str(target), 'sha256': sha(target), 'quality': q}
    except Exception as error:
        target.unlink(missing_ok=True)
        return {'ok': False, 'variant': variant, 'error': f'{type(error).__name__}: {error}'}

def candidate_signature(plan, variant):
    return digest({'topology': plan['topology'], 'nodes': sorted([
        (plan['organic_node_type'], plan['organic_sha256']),
        (plan['metal_node_type'], variant['sha256'])]), 'edges': []})

def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)

def run_substitutions(plan_path, forwarded, history, history_paths):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args(forwarded)
    if args.count < 1 or args.workers < 1:
        raise ValueError('Count and workers must be positive')
    plan_path = plan_path.resolve(); plan = json.loads(plan_path.read_text())
    assert plan['schema'] == 1 and plan['kind'] == 'hypothetical_ordered_metal_substitution'
    assert plan['allowed_substitutions'] == {'Mo': 'W', 'Zn': 'Cd'}
    for p, h in [('topology_path','topology_sha256'),('template_metal_path','template_metal_sha256'),
                 ('organic_path','organic_sha256'),('source_verified_cif','source_verified_cif_sha256')]:
        assert sha(plan[p]) == plan[h], p
    parent = BuildingBlock(plan['template_metal_path'])
    seen = set()
    for v in plan['variants']:
        assert sha(v['path']) == v['sha256']
        block = BuildingBlock(v['path'])
        assert np.array_equal(block.atoms.positions, parent.atoms.positions)
        assert np.array_equal(block.bonds, parent.bonds)
        assert all(old == new or (old, new) in [('Mo','W'),('Zn','Cd')]
                   for old,new in zip(parent.atoms.symbols, block.atoms.symbols))
        assert canonical_key(block.atoms.numbers, block.bonds) == v['graph_sha256']
        assert v['graph_sha256'] not in seen; seen.add(v['graph_sha256'])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    database = output/'generation.sqlite3'
    if any(output.iterdir()) and not database.exists():
        raise FileExistsError('Use a fresh output or the same substitution run')
    fingerprint = digest({'version': VERSION, 'plan': sha(plan_path), 'code': sha(__file__), 'history': history_paths})
    c = sqlite3.connect(database)
    c.executescript('''
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS samples(
          id INTEGER PRIMARY KEY,sample_id TEXT UNIQUE,signature TEXT UNIQUE,cif_sha256 TEXT UNIQUE,
          topology TEXT,cif_path TEXT,nodes_json TEXT,edges_json TEXT,quality_json TEXT,
          combination_signature TEXT UNIQUE,node_graph_sha256 TEXT UNIQUE,mask INTEGER,created_at REAL);
        CREATE TABLE IF NOT EXISTS attempts(mask INTEGER PRIMARY KEY,status TEXT,detail TEXT);
    ''')
    old = c.execute("select value from meta where key='fingerprint'").fetchone()
    if old and old[0] != fingerprint:
        raise ValueError('Substitution plan, code, or history changed; use a separate output')
    if not old:
        c.execute('insert into meta values(?,?)', ('fingerprint',fingerprint))
        c.execute('insert into meta values(?,?)', ('generator_kind','ordered_metal_substitution_v1'))
        c.commit()
    save_json(output/'duplicate_history.json', {'databases':history_paths,'policy':'assignment_signature_and_exact_cif_hash',
                                               'additional_uniqueness':'topology_plus_canonical_element_labeled_metal_node_graph'})
    configure_quiet_builder_logging()
    if not (output/'template/metadata.json').exists():
        print('Building and validating the unchanged mofbb template', flush=True)
        prepare_template(plan, output)
    (output/'.staging').mkdir(exist_ok=True)
    done = {r[0] for r in c.execute('select mask from attempts')}
    accepted = c.execute('select count(*) from samples').fetchone()[0]
    candidates = iter(v for v in plan['variants'] if v['mask'] not in done)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(plan,str(output))) as pool:
        while accepted < args.count:
            batch = []
            for _ in range(min(2*args.workers, args.count-accepted)):
                v = next(candidates, None)
                if v is None: break
                signature = candidate_signature(plan,v)
                if history.contains(signature=signature):
                    c.execute('insert into attempts values(?,?,?)',(v['mask'],'history_duplicate',signature));c.commit()
                    continue
                batch.append(v)
            if not batch:
                if len(done) == len(plan['variants']): break
                # No candidates remain, or an all-history-duplicate final batch.
                remaining = [v for v in plan['variants'] if not c.execute('select 1 from attempts where mask=?',(v['mask'],)).fetchone()]
                if not remaining: break
                candidates=iter(remaining); continue
            for result in pool.map(generate_one,batch,chunksize=1):
                v=result['variant']; done.add(v['mask'])
                if not result['ok']:
                    c.execute('insert into attempts values(?,?,?)',(v['mask'],'rejected',result['error']));c.commit()
                    print('rejected',v['name'],result['error'],flush=True);continue
                signature=candidate_signature(plan,v)
                if history.contains(signature, result['sha256']):
                    Path(result['path']).unlink();c.execute('insert into attempts values(?,?,?)',(v['mask'],'history_duplicate',signature));c.commit();continue
                sample_id=f"single_linker_mof_{accepted+1:07d}_{plan['topology']}"
                relative=Path('mofs')/f'{accepted//10000:05d}'/f'{sample_id}.cif'
                target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
                Path(result['path']).replace(target)
                nodes=[{'node_type':plan['organic_node_type'],'kind':'organic_node','source_path':plan['organic_path'],
                        'block_sha256':plan['organic_sha256'],'structure_id':'USER_SMILES_20260920'},
                       {'node_type':plan['metal_node_type'],'kind':'metal_node','source_path':v['path'],
                        'block_sha256':v['sha256'],'structure_id':v['name'],'hypothetical':True}]
                c.execute('insert into samples values(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                          (accepted+1,sample_id,signature,result['sha256'],plan['topology'],str(relative),json.dumps(nodes),'[]',
                           json.dumps(result['quality']),digest([plan['topology'],v['graph_sha256']]),v['graph_sha256'],v['mask'],time.time()))
                c.execute('insert into attempts values(?,?,?)',(v['mask'],'accepted',sample_id));c.commit();accepted+=1
            progress={'target':args.count,'accepted':accepted,'attempted':c.execute('select count(*) from attempts').fetchone()[0],
                      'elapsed_seconds':time.monotonic()-started,'status':'complete' if accepted>=args.count else 'running'}
            save_json(output/'progress.json',progress)
            print(f"accepted={accepted}/{args.count} attempted={progress['attempted']}",flush=True)
    rejected=c.execute("select count(*) from attempts where status='rejected'").fetchone()[0]
    summary={'status':'complete' if accepted>=args.count else 'exhausted','requested':args.count,'completed':accepted,
             'attempted':c.execute('select count(*) from attempts').fetchone()[0],'rejected':rejected,
             'unique_node_graphs':c.execute('select count(distinct node_graph_sha256) from samples').fetchone()[0],
             'topology':plan['topology'],'hypothetical':True,'organic_connection_sites':3,'energy_relaxed':False,
             'charge_balance_validated':False,'fingerprint':fingerprint,'plan':str(plan_path)}
    save_json(output/'generation_summary.json',summary)
    c.close();print(json.dumps(summary,indent=2),flush=True)
    if accepted<args.count:raise RuntimeError(f'Only {accepted} of {args.count} substitutions passed; see generation_summary.json')
