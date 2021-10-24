#!/usr/bin/env python3
"""Dependency Graph Traversal."""
from __future__ import annotations

import concurrent.futures
import dataclasses
import enum
import threading
import time

from collections.abc import Set, Hashable, Collection, Mapping
from typing import Optional, Callable, Any

@dataclasses.dataclass
class SubNet:
    """A representation of a subnet"""
    upper: set[Hashable]
    nodes: set[Hashable]
    lower: set[Hashable]

class Direction(enum.Enum):
    """Possible directions for graph traversal."""
    UP = 0
    DN = 1

    def invert(self) -> Direction:
        """Returns the inverse of the Direction."""
        return Direction((self.value + 1) % 2)

class Graph:
    """Represent directional edges among nodes of any immmutable type."""

    _lock: threading.RLock
    _nodes: dict[Hashable, tuple[set[Hashable], set[Hashable]]]

    def __init__(self):
        """Initialise the new graph."""
        self._lock = threading.RLock()
        self._nodes = {}        # map node to its edges

    def _neighbours(self, n: Hashable, direction: Direction) -> set[Hashable]:
        """Get N's neighbours in DIRECTION."""
        result = set()
        with self._lock:
            if self.has_node(n):
                result |= self._nodes[n][direction.value]
        return result

    def size(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)

    def all_nodes(self):
        """Return all nodes; order is not specified."""
        with self._lock:
            return set(self._nodes)

    def _valid(self, nodes: Optional[Collection[Hashable]]) -> bool:
        """Return a subset of NODES actually in the graph."""
        with self._lock:
            if nodes is None:
                return set()
            else:
                return set(n for n in nodes if n in self._nodes)

    def has_node(self, n: Hashable) -> bool:
        """Check if N is a node in the graph."""
        with self._lock:
            return n in self._nodes

    def add_node(self, n: Hashable):
        """Add N as a graph node; no effect if already present."""
        with self._lock:
            if not self.has_node(n):
                self._nodes[n] = (set(), set())

    def predecessors(self, n: Hashable) -> set[Hashable]:
        """Return N's immediate upstream neighbours."""
        return self._neighbours(n, Direction.UP)

    def successors(self, n: Hashable) -> set[Hashable]:
        """Return N's immediate downstream neighbours."""
        return self._neighbours(n, Direction.DN)

    def _neighbourless(self, direction: Direction) -> set[Hashable]:
        """Return nodes without neighbours in DIRECTION."""
        with self._lock:
            return set(n for n in self._nodes
                       if not self._nodes[n][direction.value])

    def highest(self) -> set[Hashable]:
        """Return nodes without predecessors."""
        return self._neighbourless(Direction.UP)

    def lowest(self) -> set[Hashable]:
        """Return nodes without successors."""
        return self._neighbourless(Direction.DN)

    def _remove_neighbour_references(self, n: Hashable, direction: Direction):
        """Remove references to N from N's neighbours in DIRECTION."""
        with self._lock:
            vecinos = self._neighbours(n, direction)
            reverse = direction.invert()
            for v in vecinos:
                self._nodes[v][reverse.value].remove(n)

    def del_node(self, n: Hashable):
        """Remove N from the graph."""
        with self._lock:
            self._remove_neighbour_references(n, Direction.UP)
            self._remove_neighbour_references(n, Direction.DN)
            self._nodes[n][Direction.UP.value].clear()
            self._nodes[n][Direction.DN.value].clear()
            del self._nodes[n]

    def add_edge(self, hi: Hashable, lo: Hashable):
        """Add an edge from HI to LO."""
        self.add_node(hi)
        self.add_node(lo)
        with self._lock:
            self._nodes[hi][Direction.DN.value].add(lo)
            self._nodes[lo][Direction.UP.value].add(hi)

    def del_edge(self, hi: Hashable, lo: Hashable):
        """Remove any existing edge from HI to LO."""
        with self._lock:
            if hi in self._nodes and lo in self._nodes:
                self._nodes[hi][Direction.DN.value].discard(lo)
                self._nodes[lo][Direction.UP.value].discard(hi)

    def _reachables(self,
                    starts: Optional[Collection[Hashable]],
                    direction: Direction,
                    stops: Optional[Collection[Hashable]]) -> set[Hashable]:
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

    def upstreams(self, n: Hashable) -> set[Hashable]:
        """Return all nodes upstream from N."""
        return self._reachables([n], Direction.UP, None)

    def downstreams(self, n: Hashable) -> set[Hashable]:
        """Return all nodes downstream from N."""
        return self._reachables([n], Direction.DN, None)

    def subnet(self,
               request_hi: Optional[Collection[Hashable]]=None,
               request_lo: Optional[Collection[Hashable]]=None) -> SubNet:
        """Return subnet spanning REQUEST_HI to REQUEST_LO nodes."""
        start_hi = self.highest() if request_hi is None else request_hi
        start_lo = self.lowest() if request_lo is None else request_lo
        fan_dn = self._reachables(start_hi, Direction.DN, request_lo)
        fan_up = self._reachables(start_lo, Direction.UP, request_hi)
        node_set = fan_dn & fan_up
        return SubNet(upper=set(e for e in start_hi if e in node_set),
                      nodes=node_set,
                      lower=set(e for e in start_lo if e in node_set))

    def subnet_edges(self, subnet: SubNet) -> list[tuple[Hashable, Hashable]]:
        """Return the edges in the SUBNET."""
        result = []
        if isinstance(subnet, SubNet):
            visited = set()
            pending = set(subnet.upper)
            while pending:
                cur = pending.pop()
                for nxt in (n for n in self.successors(cur)
                            if n in subnet.nodes):
                    edge = (cur, nxt)
                    result.append(edge)
                    if nxt not in visited:
                        pending.add(nxt)
                visited.add(cur)
        return result

class NonTerminatingException(Exception):
    pass

class Runner:
    """Runs over the graph."""
    _net: Graph
    _op: Callable[[Hashable], Any]
    _lock: threading.RLock
    _rspec_graph: Optional[SubNet]
    _rspec_meta: dict[Hashable, Any]
    _rv_pool: concurrent.futures.Executor
    _rv_stop: threading.Event
    _rv_enter: dict[Hashable, concurrent.futures.Future]
    _rv_leave: dict[Hashable, time.time]
    _rv_block: set[Hashable]
    def __init__(self, graph: Graph, operator):
        self._net = graph
        self._op = operator
        self._lock = threading.RLock()
        self._rspec_graph = None
        self._rspec_meta = {}
        self._rv_pool = None
        self._rv_stop = threading.Event()
        self._rv_enter = {}
        self._rv_leave = {}
        self._rv_block = set()

    def completions(self) -> set[Hashable]:
        with self._lock:
            return set(self._rv_leave.keys())

    def blocked(self) -> set[Hashable]:
        with self._lock:
            return set(self._rv_block)

    def result(self, node: Hashable) -> Optional[Any]:
        outcome = None
        with self._lock:
            future = self._rv_enter.get(node, None)
            outcome = future.result() if future and future.done() else None
        return outcome

    def run(self, skeins: int, timeout_limit: float,
            request_hi: Optional[Collection[Hashable]],
            request_lo: Optional[Collection[Hashable]],
            meta: Mapping[Hashable, Any]={}):
        with self._lock:
            self._rv_stop.clear()
            self._rv_enter.clear()
            self._rv_leave.clear()
            self._rv_block.clear()
        if self._op and self._net and self._net.size():
            with self._lock:
                self._rspec_meta = dict(meta)
                self._rspec_graph = self._net.subnet(request_hi, request_lo)
            with concurrent.futures.ThreadPoolExecutor(max_workers=skeins) as self._rv_pool:
                self._visit_multi(self._rspec_graph.upper)
                self._wait_done(timeout_limit)


    def _wait_done(self, timeout_limit: float):
        self._rv_stop.wait(timeout_limit)

    def _visit_multi(self, nodes_to_visit: Collection[Hashable]):
        if not nodes_to_visit:
            return
        for n in (v for v in nodes_to_visit if v in self._rspec_graph.nodes):
            if self._rv_stop.is_set():
                continue
            prereqs = set(v for v in self._net.predecessors(n)
                          if v in self._rspec_graph.nodes)
            with self._lock:
                completions = set(self._rv_leave.keys())
                if completions >= prereqs and n not in self._rv_enter:
                    future = self._rv_pool.submit(self._visit_single, n,
                                                  **self._rspec_meta)
                    self._rv_enter[n] = future

    def _visit_single(self, node: Hashable, **kwargs: Mapping[Hashable, Any]) -> Any:
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

    def _update_blocks(self, node: Hashable):
        """Update the block set for NODE."""
        with self._lock:
            self._rv_block |= (self._rspec_graph.nodes & self._net.downstreams(node))
                

# Local Variables:
# mode: python
# python-indent: 4
# End:
# vim: set expandtab:
