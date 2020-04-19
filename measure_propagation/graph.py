import logging
from queue import PriorityQueue

import numpy as np

logger = logging.getLogger(__name__)


class Graph(dict):
    """
    Implements Adjacency List for graph representation extending the Python's dict.
    """
    NEIGHBOURS_KEY = 'neighbours'
    WEIGHTS_KEY = 'weights'

    @property
    def vertices(self):
        return self.keys()

    @property
    def nb_vertices(self):
        return len(self)

    def add_edge(self, vertex, neighbour, weight=1):
        if vertex not in self:
            self[vertex] = {self.NEIGHBOURS_KEY: [neighbour],
                            self.WEIGHTS_KEY: [weight]
                            }
        elif neighbour not in self[vertex][self.NEIGHBOURS_KEY]:
            self[vertex][self.NEIGHBOURS_KEY].append(neighbour)
            self[vertex][self.WEIGHTS_KEY].append(weight)
        else:
            logger.debug("Vertex %d is already in the neighbour list" % neighbour)

    def build(self, *args):
        self._build(*args)
        self.post_build_hook()
        return self

    def _build(self, edges):
        for source_id, dest_id in edges:
            if source_id != dest_id:
                self.add_edge(source_id, dest_id)
                self.add_edge(dest_id, source_id)

    def post_build_hook(self):
        for vertex, edge_info in self.items():
            edge_info[self.WEIGHTS_KEY] = np.expand_dims(np.array(edge_info[self.WEIGHTS_KEY]), 1)


class KNNGraph(Graph):
    """
    Implementation of the k-nearest neighbor graph.
    """
    def __init__(self, K, similarity_func, *args, **kwargs):
        super(KNNGraph, self).__init__(*args, **kwargs)
        self.K = K
        self.similarity_func = similarity_func

    class Neighbour:
        def __init__(self, id_, similarity):
            self.id = id_
            self.similarity = similarity

        def __lt__(self, other):
            return self.similarity < other.similarity

    def _build(self, X):
        n_samples = len(X)
        for i in range(n_samples):
            # A priority queue is used in order to mantain the K-most similar vertices
            neighbours = PriorityQueue(maxsize=self.K)
            for j in range(n_samples):
                if i != j:
                    neighbour = KNNGraph.Neighbour(j, self.similarity_func(X[i], X[j]))
                    if not neighbours.full():
                        neighbours.put(neighbour)
                    else:
                        lowest_entry = neighbours.get()
                        neighbours.put(neighbour) if neighbour.similarity > lowest_entry.similarity else neighbours.put(
                            lowest_entry)
            # create edges
            while not neighbours.empty():
                neighbour = neighbours.get()
                self.add_edge(i, neighbour.id, weight=neighbour.similarity)
