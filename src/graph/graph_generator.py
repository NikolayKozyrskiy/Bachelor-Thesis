import random

from src.graph.graph import Graph
import numpy as np


# TODO make new generator with non equal quality of nodes in each cluster
class GraphGenerator(object):
    def __init__(self, n, clusters, p_in, p_out):
        self.n = n
        self.clusters = clusters
        self.p_in = p_in
        self.p_out = p_out

    def generate_graph(self):
        nodes_in_cluster = self.n // self.clusters
        nodes_in_last_cluster = nodes_in_cluster + self.n % self.clusters

        nodes = []
        for i in range(0, self.clusters - 1):
            nodes.extend([i] * nodes_in_cluster)
        nodes.extend([self.clusters - 1] * nodes_in_last_cluster)
        random.shuffle(nodes)

        edges = np.zeros((self.n, self.n))

        random_pin = np.random.choice([0, 1], edges.shape, p=[1 - self.p_in, self.p_in])
        random_pout = np.random.choice([0, 1], edges.shape, p=[1 - self.p_out, self.p_out])

        for i in range(0, self.n):
            for j in range(0, self.n):
                is_same_cluster = nodes[i] == nodes[j]

                if is_same_cluster:
                    edges[i][j] = random_pin[i][j]
                else:
                    edges[i][j] = random_pout[i][j]

        return Graph(self.n, nodes, edges)
