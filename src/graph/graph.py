class Graph():
    def __init__(self, n, nodes, edges):
        self.n = n
        self.edges = edges
        self.nodes = nodes

    def __str__(self):
        res = "nodes: \n"
        for k in self.nodes:
            res += str(k) + " "
        res += "\nedges:\n"
        for edge in self.edges:
            res += str(edge) + " \n"
        return res