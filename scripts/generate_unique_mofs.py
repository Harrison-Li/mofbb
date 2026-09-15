#!/usr/bin/env python3
"""Launch the unchanged bulk generator with cross-dataset duplicate exclusion.

All ordinary bulk-generator flags are forwarded. --history can be repeated;
by default it uses the organized 2m_mofs_v2 database, including archived PCU
history. This rejects assignment-signature and exact-CIF-hash duplicates, not
all crystallographically equivalent structures.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

import generate_500k_single_linker_mofs as bulk

ENV = 'MOFBB_DUPLICATE_HISTORY_DATABASES'
DEFAULT_HISTORY = bulk.ROOT / '2m_mofs_v2' / 'generation.sqlite3'
_original_candidate = bulk.candidate_at
_original_init = bulk.init_build_worker
_original_persist = bulk.persist
_original_connect = bulk.connect
_index = None


class HistoryIndex:
    def __init__(self, paths):
        self.paths = [Path(p).resolve() for p in paths]
        self.connections = []
        self.pid = None

    def close(self):
        for connection, _ in self.connections:
            connection.close()
        self.connections = []
        self.pid = None

    def open(self):
        if self.pid == os.getpid():
            return
        self.close()
        pending = list(self.paths)
        visited = set()
        for path in pending:
            path = path.resolve()
            if path in visited:
                continue
            visited.add(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=60)
            tables = {r[0] for r in connection.execute("select name from sqlite_master where type='table'")}
            if 'samples' not in tables:
                connection.close()
                raise ValueError(f'Not a generation database: {path}')
            selected = ['samples']
            if 'generation_history_samples' in tables:
                selected.append('generation_history_samples')
            for table in selected:
                connection.execute(f'select signature,cif_sha256 from {table} limit 0')
            self.connections.append((connection, selected))
            # Each imported run retains its own full database and candidate state.
            # Follow registered archives without mixing their generator cursors.
            if 'generation_history_databases' in tables:
                pending.extend(Path(row[0]) for row in connection.execute(
                    'select database_path from generation_history_databases order by source_dataset'
                ))
        self.pid = os.getpid()

    def contains(self, signature=None, cif_hash=None):
        self.open()
        for connection, tables in self.connections:
            for table in tables:
                if signature is not None and connection.execute(
                    f'select 1 from {table} where signature=? limit 1', (signature,)
                ).fetchone():
                    return True
                if cif_hash is not None and connection.execute(
                    f'select 1 from {table} where cif_sha256=? limit 1', (cif_hash,)
                ).fetchone():
                    return True
        return False


def history():
    global _index
    if _index is None:
        _index = HistoryIndex(json.loads(os.environ[ENV]))
    return _index


def candidate_at(*args, **kwargs):
    candidate = _original_candidate(*args, **kwargs)
    if candidate is not None and history().contains(signature=candidate.signature):
        return None
    return candidate


def init_build_worker(config):
    # Also installs the hook in spawned workers, not only forked workers.
    _original_init(config)
    history().open()
    bulk.candidate_at = candidate_at


def persist(connection, output, result, *args, **kwargs):
    accepted = []
    for success in result['successes']:
        if history().contains(success['candidate'].signature, success['quality']['cif_sha256']):
            Path(success['cif']).unlink(missing_ok=True)
        else:
            accepted.append(success)
    filtered = dict(result, successes=accepted)
    if len(accepted) != len(result['successes']):
        print(f"history_duplicates_rejected={len(result['successes']) - len(accepted)}", flush=True)
    return _original_persist(connection, output, filtered, *args, **kwargs)


def connect(path):
    connection = _original_connect(path)
    (Path(path).parent / 'duplicate_history.json').write_text(json.dumps({
        'databases': json.loads(os.environ[ENV]),
        'policy': 'assignment_signature_before_build_and_signature_or_exact_hash_before_save',
        'includes_removed_generation_history': True,
    }, indent=2) + '\n')
    return connection


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--history', type=Path, action='append')
    parser.add_argument('--check-history', action='store_true')
    args, forwarded = parser.parse_known_args()
    paths = []
    for path in args.history or [DEFAULT_HISTORY]:
        path = path.expanduser().resolve()
        paths.append(str(path / 'generation.sqlite3' if path.is_dir() else path))
    os.environ[ENV] = json.dumps(list(dict.fromkeys(paths)))
    index = history()
    index.open()
    if args.check_history:
        print(json.dumps({'status': 'ready', 'databases': paths,
                          'tables': [tables for _, tables in index.connections]}, indent=2))
        return
    if '--help' not in forwarded and '-h' not in forwarded:
        output_parser = argparse.ArgumentParser(add_help=False)
        output_parser.add_argument('--output', type=Path, required=True)
        output_args, _ = output_parser.parse_known_args(forwarded)
        output = output_args.output.expanduser().resolve()
        if str(output / 'generation.sqlite3') in paths:
            raise ValueError('Use a separate output folder; history databases are read-only inputs.')
        ledger = output / 'duplicate_history.json'
        if ledger.exists() and json.loads(ledger.read_text())['databases'] != json.loads(os.environ[ENV]):
            raise ValueError('History inputs changed on resume; keep the original history or use a fresh output.')
        if (output / 'generation.sqlite3').exists() and not ledger.exists():
            raise ValueError('Existing output was not started by this launcher; use a fresh output.')
    bulk.init_build_worker = init_build_worker
    bulk.candidate_at = candidate_at
    bulk.persist = persist
    bulk.connect = connect
    sys.argv = [str(Path(bulk.__file__)), *forwarded]
    try:
        bulk.main()
    finally:
        index.close()


if __name__ == '__main__':
    main()
