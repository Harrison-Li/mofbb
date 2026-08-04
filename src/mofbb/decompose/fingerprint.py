"""Connectivity-aware fingerprinting for decomposition fragments."""

from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha1

import ase
import networkx as nx


def fragment_fingerprint(
    symbols: list[str], bonds: Iterable[tuple[int, int, str]]
) -> str:
    """Return a stable connectivity-aware fragment hash."""

    graph = nx.Graph()
    for i, symbol in enumerate(symbols):
        graph.add_node(i, symbol=symbol)
    for i, j, _ in bonds:
        graph.add_edge(i, j)

    wl_hash = nx.weisfeiler_lehman_graph_hash(graph, node_attr="symbol")
    payload = {
        "formula": ase.Atoms(symbols=symbols).get_chemical_formula(),
        "wl_hash": wl_hash,
    }
    return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
