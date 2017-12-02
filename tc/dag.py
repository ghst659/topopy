#!/usr/bin/env python3
# Copyright (c) 2016 ghst659@github.com
# All rights reserved.
"""Dependency Graph Traversal."""
from __future__ import print_function

import collections
import concurrent.futures
import threading
import time
##############################################################################
SubNet = collections.namedtuple("SubNet", field_names=["upper", "nodes", "lower"])

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
        with self._lock:
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

    def subnet(self, request_hi=None, request_lo=None):
        """Return subnet spanning REQUEST_HI to REQUEST_LO nodes."""
        start_hi = self.highest() if request_hi is None else request_hi
        start_lo = self.lowest() if request_lo is None else request_lo
        fan_dn = self._reachables(start_hi, self.DN, request_lo)
        fan_up = self._reachables(start_lo, self.UP, request_hi)
        node_set = fan_dn & fan_up
        return SubNet(upper=set(e for e in start_hi if e in node_set),
                      nodes=node_set,
                      lower=set(e for e in start_lo if e in node_set))

    def subnet_edges(self, subnet_tuple):
        """Return the edges in the SUBNET_TUPLE."""
        result = []
        if isinstance(subnet_tuple, SubNet):
            visited = set()
            pending = set(subnet_tuple.upper)
            while pending:
                cur = pending.pop()
                for nxt in (n for n in self.successors(cur)
                            if n in subnet_tuple.nodes):
                    edge = (cur, nxt)
                    result.append(edge)
                    if nxt not in visited:
                        pending.add(nxt)
                visited.add(cur)
        return result

class NonTerminatingException(Exception):
    pass

class Runner:
    def __init__(self, graph, operator):
        self._net = graph
        self._op = operator
        self._lock = threading.RLock()
        self._rspec_graph = None
        self._rspec_op = None
        self._rv_pool = None
        self._rv_stop = threading.Event()
        self._rv_enter = {}
        self._rv_leave = {}
        self._rv_block = set()

    def completions(self):
        with self._lock:
            return set(self._rv_leave.keys())

    def blocked(self):
        with self._lock:
            return set(self._rv_block)

    def result(self, node):
        outcome = None
        with self._lock:
            future = self._rv_enter.get(node, None)
            outcome = future.result() if future and future.done() else None
        return outcome

    def run(self, workers, timeout_limit, request_hi, request_lo, meta={}):
        with self._lock:
            self._rv_stop.clear()
            self._rv_enter.clear()
            self._rv_leave.clear()
            self._rv_block.clear()
        if self._op and self._net and self._net.size():
            with self._lock:
                self._rspec_meta = dict(meta)
                self._rspec_graph = self._net.subnet(request_hi, request_lo)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as self._rv_pool:
                self._visit_multi(self._rspec_graph.upper)
                self._wait_done(timeout_limit)


    def _wait_done(self, timeout_limit):
        self._rv_stop.wait(timeout_limit)

    def _visit_multi(self, nodes_to_visit):
        if nodes_to_visit:
            for n in (v for v in nodes_to_visit if v in self._rspec_graph.nodes):
                if not self._rv_stop.is_set():
                    prereqs = set(v for v in self._net.predecessors(n)
                                  if v in self._rspec_graph.nodes)
                    with self._lock:
                        completions = set(self._rv_leave.keys())
                        if completions >= prereqs and n not in self._rv_enter:
                            future = self._rv_pool.submit(self._visit_single, n,
                                                            **self._rspec_meta)
                            self._rv_enter[n] = future

    def _visit_single(self, node, **kwargs):
        result = None
        try:
            result = self._op.visit(node, **kwargs)
        except NonTerminatingException:
            self._update_blocks(node)
            raise
        except:
            self._update_blocks(node)
            self._rv_stop.set()
            raise
        finally:
            self._rv_leave[node] = time.time()
        with self._lock:
            concluded = self._rv_block | set(self._rv_leave.keys())
            if concluded >= self._rspec_graph.nodes:
                self._rv_stop.set()
                # self._rv_pool.shutdown()
            else:
                self._visit_multi(self._net.successors(node))
        return result

    def _update_blocks(self, node):
        """Update the block set for NODE."""
        with self._lock:
            self._rv_block |= (self._rspec_graph.nodes & self._net.downstreams(node))
                

# Local Variables:
# mode: python
# python-indent: 4
# End:
# vim: set expandtab:
