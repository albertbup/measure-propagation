import logging

import numpy as np

logger = logging.getLogger(__name__)


class MeasurePropagation:
    """
    Implements Measure Propagation algorithm.
    """
    def __init__(self, mu=0.1, nu=0.01, tol=2e-2, max_iter=100):
        self.graph = None
        self.r, self.nb_classes = None, None
        self.p, self.q = None, None
        self.mu = mu
        self.nu = nu
        self.tol = tol
        self.max_iter = max_iter
        self.SMALL = 1e-10  # to ensure that we never take log(0)

    def _labels_to_probabilities(self, vertices_labels_dct):
        nb_classes = len(np.unique(list(vertices_labels_dct.values())))
        probs = {}
        for vertex, label in vertices_labels_dct.items():
            probs[vertex] = np.zeros(nb_classes)
            probs[vertex][label] = 1
        return probs, nb_classes

    def _init_probability_distributions(self):
        p = np.full((self.graph.nb_vertices, self.nb_classes), 1 / self.nb_classes)
        q = np.full((self.graph.nb_vertices, self.nb_classes), 1 / self.nb_classes)
        return p, q

    def compute_p_update(self, vertex):
        neighbours = self.graph[vertex][self.graph.NEIGHBOURS_KEY]
        w_neighbours = self.graph[vertex][self.graph.WEIGHTS_KEY]
        gamma = self.nu + self.mu * w_neighbours.sum()
        p_new = np.exp((np.log(self.q[neighbours] + self.SMALL) * w_neighbours).sum(axis=0) * (self.mu / gamma))
        return p_new / p_new.sum()

    def compute_p_updates(self):
        for vertex in self.graph.vertices:
            self.p[vertex] = self.compute_p_update(vertex)

    def compute_q_update(self, vertex):
        neighbours = self.graph[vertex][self.graph.NEIGHBOURS_KEY]
        w_neighbours = self.graph[vertex][self.graph.WEIGHTS_KEY]
        divident_right_sum = self.mu * (w_neighbours * self.p[neighbours]).sum(axis=0)
        divident_left_sum = self.r[vertex] if vertex in self.r else 0
        divisor_right_sum = self.mu * w_neighbours.sum()
        divisor_left_sum = 1 if vertex in self.r else 0
        return (divident_right_sum + divident_left_sum) / (divisor_right_sum + divisor_left_sum)

    def compute_q_updates(self):
        q_new = np.zeros(self.q.shape)
        for vertex in self.graph.vertices:
            q_new[vertex] = self.compute_q_update(vertex)
        return q_new

    def alternate_minimization_step(self):
        self.compute_p_updates()
        q_new = self.compute_q_updates()
        test_convergence = self.compute_test_convergence(q_new)
        logger.debug("Test convergence %f" % test_convergence)
        self.q = q_new
        return test_convergence

    def compute_test_convergence(self, q_new):
        div = q_new / (self.q + self.SMALL)
        beta = np.log(np.max(div, 1) + self.SMALL)
        accum = .0
        for vertex in self.graph.vertices:
            delta = 1 if vertex in self.r else 0
            d_i = np.array(self.graph[vertex][self.graph.WEIGHTS_KEY]).sum()
            accum += (delta + d_i) * beta[vertex]
        return accum

    def optimize(self, graph, vertices_labels_dct):
        self.graph = graph
        self.r, self.nb_classes = self._labels_to_probabilities(vertices_labels_dct)
        self.p, self.q = self._init_probability_distributions()

        convergences = []
        for it in range(self.max_iter):
            logger.debug("Iteration %d" % it)
            convergences.append(self.alternate_minimization_step())
            if it > 0:
                change = (convergences[it - 1] - convergences[it]) / convergences[it]
                logger.debug("Change in convergence criteria %f" % change)
                if change <= self.tol:
                    break

        logger.info("Convergence criterion reached on iteration %d" % it)
        logger.info("Relative objective value: %f", convergences[it])

    def get_output_labels(self):
        return np.argmax(self.q, axis=1)
