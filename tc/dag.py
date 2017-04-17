#!/usr/bin/env python3
# Copyright (c) 2016 ghst659@github.com
# All rights reserved.
"""Dependency Graph Traversal."""
from __future__ import print_function

import collections
import threading
##############################################################################
SubGraph = collections.namedtuple("SubGraph", field_names=["upper", "nodes", "lower"])

class Graph:
    """Represent directional edges among nodes of any immmutable type."""

    UP = 0
    DN = 1

    @classmethod
    def invert(cls, direction):
        """Return the reverse of DIRECTION."""
        return (direction + 1) % 2

    def __init__(self):
        """Initialise the new graph."""
        self._lock = threading.RLock()
        self._nodes = {}        # map node to its edges

    def _neighbours(self, n, direction):
        """Get N's neighbours in DIRECTION."""
        result = set()
        with self._lock:
            if self.has_node(n):
                result |= self._nodes[n][direction]
        return result

    def size(self):
        """Return the number of nodes."""
        return len(self._nodes)

    def all_nodes(self):
        """Return all nodes; order is not specified."""
        return set(self._nodes)

    def _valid(self, nodes):
        """Return a subset of NODES actually in the graph."""
        with self._lock:
            if nodes is None:
                return set()
            else:
                return set(n for n in nodes if n in self._nodes)

    def has_node(self, n):
        """Check if N is a node in the graph."""
        with self._lock:
            return n in self._nodes

    def add_node(self, n):
        """Add N as a graph node; no effect if already present."""
        with self._lock:
            if not self.has_node(n):
                self._nodes[n] = (set(), set())

    def predecessors(self, n):
        """Return N's immediate upstream neighbours."""
        return self._neighbours(n, self.UP)

    def successors(self, n):
        """Return N's immediate downstream neighbours."""
        return self._neighbours(n, self.DN)

    def _neighbourless(self, direction):
        """Return nodes without neighbours in DIRECTION."""
        with self._lock:
            return set(n for n in self._nodes if not self._nodes[n][direction])

    def highest(self):
        """Return nodes without predecessors."""
        return self._neighbourless(self.UP)

    def lowest(self):
        """Return nodes without successors."""
        return self._neighbourless(self.DN)

    def _remove_neighbour_references(self, n, direction):
        """Remove references to N from N's neighbours in DIRECTION."""
        with self._lock:
            vecinos = self._neighbours(n, direction)
            reverse = self.invert(direction)
            for v in vecinos:
                self._nodes[v][reverse].remove(n)

    def del_node(self, n):
        """Remove N from the graph."""
        with self._lock:
            self._remove_neighbour_references(n, self.UP)
            self._remove_neighbour_references(n, self.DN)
            self._nodes[n][self.UP].clear()
            self._nodes[n][self.DN].clear()
            del self._nodes[n]

    def add_edge(self, hi, lo):
        """Add an edge from HI to LO."""
        self.add_node(hi)
        self.add_node(lo)
        with self._lock:
            self._nodes[hi][self.DN].add(lo)
            self._nodes[lo][self.UP].add(hi)

    def del_edge(self, hi, lo):
        """Remove any existing edge from HI to LO."""
        with self._lock:
            if hi in self._nodes and lo in self._nodes:
                self._nodes[hi][self.DN].discard(lo)
                self._nodes[lo][self.UP].discard(hi)

    def _reachables(self, starts, direction, stops):
        """Trace reachable nodes from STARTS in DIRECTION to STOPS."""
        result = set()
        if stops is None:
            stops = []
        with self._lock:
            visited = set()
            pending = set(n for n in starts if n in self._nodes)
            while pending:
                cur = pending.pop()
                result.add(cur)
                if cur not in stops:
                    for nxt in (v for v in self._neighbours(cur, direction)
                                if v not in visited):
                        pending.add(nxt)
                visited.add(cur)
        return result

    def upstreams(self, n):
        """Return all nodes upstream from N."""
        return self._reachables([n], self.UP, None)

    def downstreams(self, n):
        """Return all nodes downstream from N."""
        return self._reachables([n], self.DN, None)

    def subgraph(self, request_hi=None, request_lo=None):
        """Return subgraph spanning REQUEST_HI to REQUEST_LO nodes."""
        start_hi = self.highest() if request_hi is None else request_hi
        start_lo = self.lowest() if request_lo is None else request_lo
        fan_dn = self._reachables(start_hi, self.DN, request_lo)
        fan_up = self._reachables(start_lo, self.UP, request_hi)
        return SubGraph(upper=start_hi, nodes=fan_dn & fan_up, lower=start_lo)

    def subgraph_edges(self, subgraph_tuple):
        """Return the edges in the SUBGRAPH_TUPLE."""
        result = []
        if isinstance(subgraph_tuple, SubGraph):
            visited = set()
            pending = set(subgraph_tuple.upper)
            while pending:
                cur = pending.pop()
                for nxt in (n for n in self.successors(cur)
                            if n in subgraph_tuple.nodes):
                    edge = (cur, nxt)
                    result.append(edge)
                    if nxt not in visited:
                        pending.add(nxt)
                visited.add(cur)
        return result

# Local Variables:
# mode: python
# python-indent: 4
# End:
# vim: set expandtab:
