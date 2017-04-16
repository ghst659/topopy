#!/usr/bin/env python3

import unittest
import tc.dag

class TestBase(unittest.TestCase):
    def setUp(self):
        self.dag = tc.dag.Graph()

    def tearDown(self):
        del self.dag

class TestGraphNodes(TestBase):
    def test_empty(self):
        self.assertEqual(0, self.dag.size())
        self.assertEqual(set(), self.dag.all_nodes())

    def test_add_nodes(self):
        nodes = [ "One", "Two", "Three" ]
        for n in nodes:
            self.dag.add_node(n)
        self.assertEqual(len(nodes), self.dag.size())
        for n in nodes:
            self.assertTrue(self.dag.has_node(n))
        self.assertEqual(set(nodes), self.dag.all_nodes())

    def test_del_nodes(self):
        nodes = [ "One", "Two", "Three" ]
        dels = ["Two"]
        for n in nodes:
            self.dag.add_node(n)
        for n in dels:
            self.dag.del_node(n)
        self.assertEqual(len(nodes) - len(dels), self.dag.size())
        for n in dels:
            self.assertFalse(self.dag.has_node(n))

class TestGraphEdges(TestBase):
    @classmethod
    def ipow(cls, x, power):
        result = 1
        for i in range(power):
            result *= x
        return result

    @classmethod
    def node_suffix(cls, depth, j):
        return "-".join([str(depth), str(j)])

    @classmethod
    def gen_pyramid(cls, node_depth, fanout, expanding):
        """Yield edges forming a pyramid"."""
        prefix = "U" if expanding else "L"
        for depth in range(1, node_depth):
            node_prefix = "X" if (depth + 1) == node_depth else prefix
            row_count = cls.ipow(fanout, depth)
            for j in range(row_count):
                tag_current = node_prefix + cls.node_suffix(depth, j)
                tag_parent = prefix + cls.node_suffix(depth-1, j // fanout)
                edge = (tag_parent, tag_current) if expanding \
                       else (tag_current, tag_parent)
                yield edge

    @classmethod
    def gen_diamond(cls, node_depth, fanout):
        for direction in (True, False):
            yield from cls.gen_pyramid(node_depth, fanout, direction)

    def test_one_edge(self):
        edge = ("One", "Two")
        self.dag.add_edge(edge[0], edge[1])
        self.assertTrue(self.dag.has_node(edge[0]))
        self.assertTrue(self.dag.has_node(edge[1]))
        self.assertEqual(set([edge[0]]), self.dag.highest())
        self.assertEqual(set([edge[1]]), self.dag.lowest())
        self.assertEqual(set([edge[1]]), self.dag.successors(edge[0]))
        self.assertEqual(set([edge[0]]), self.dag.predecessors(edge[1]))

    def test_pyramid(self):
        for edge in self.gen_pyramid(3, 2, True):
            self.dag.add_edge(*edge)
        self.assertEqual(set(["U0-0"]), self.dag.highest())
        self.assertEqual(set(["X2-0", "X2-1", "X2-2", "X2-3"]), self.dag.lowest())

    def test_diamond(self):
        for edge in self.gen_diamond(8, 2):
            self.dag.add_edge(*edge)
        self.assertEqual(set(["U0-0"]), self.dag.highest())
        self.assertEqual(set(["L0-0"]), self.dag.lowest())

class TestGraphSubGraph(TestGraphEdges):
    def test_whole(self):
        count = 0
        for edge in self.gen_diamond(3, 2):
            self.dag.add_edge(*edge)
            count += 1
        subgraph = self.dag.subgraph()
        self.assertEqual(self.dag.size(), len(subgraph.nodes))
        self.assertEqual(1, len(subgraph.upper))
        self.assertEqual(1, len(subgraph.lower))
        edges = self.dag.subgraph_edges(subgraph)
        self.assertEqual(count, len(edges))

    def test_partial(self):
        count = 0
        for edge in self.gen_diamond(3, 2):
            self.dag.add_edge(*edge)
            count += 1
        subgraph = self.dag.subgraph(["U0-0"], ["L1-1"])
        expected = set([
            "U0-0", "U1-1", "X2-2", "X2-3", "L1-1"
        ])
        self.assertEqual(expected, subgraph.nodes)
        self.assertEqual(1, len(subgraph.upper))
        self.assertEqual(1, len(subgraph.lower))
        edges = self.dag.subgraph_edges(subgraph)
        expected_edges = set([
            ("U0-0", "U1-1"),
            ("U1-1", "X2-2"),
            ("U1-1", "X2-3"),
            ("X2-2", "L1-1"),
            ("X2-3", "L1-1")
        ])
        self.assertEqual(expected_edges, set(edges))

if __name__ == "__main__":
    unittest.main()