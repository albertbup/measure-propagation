import csv
import logging

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, ParameterGrid

from measure_propagation.graph import Graph
from measure_propagation.wrapper import MeasurePropagationSklearn

logging.basicConfig(level=logging.INFO)

LABEL_ID_MAP = {"course": 0,
                "faculty": 1,
                "student": 2,
                "project": 3,
                "staff": 4}


def read_edges(city_name, folder="WebKB"):
    counter_id = 0
    vertex_id_map = {}
    edges = []

    with open(folder + "/" + city_name + ".cites") as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for source_vertex, dest_vertex in reader:
            if source_vertex != dest_vertex:
                if source_vertex not in vertex_id_map:
                    vertex_id_map[source_vertex] = counter_id
                    counter_id += 1
                if dest_vertex not in vertex_id_map:
                    vertex_id_map[dest_vertex] = counter_id
                    counter_id += 1
                edges.append((vertex_id_map[source_vertex], vertex_id_map[dest_vertex]))

    return edges, vertex_id_map


def read_labels(city_name, vertex_id_map, folder="WebKB"):
    labels = np.zeros((len(vertex_id_map),), dtype=np.int32)
    with open(folder + "/" + city_name + ".content") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            vertex, label_str = row[0], row[-1]
            if vertex in vertex_id_map:
                labels[vertex_id_map[vertex]] = LABEL_ID_MAP[label_str]
    return labels


def main():
    city_name = "wisconsin"
    edges, vertex_id_map = read_edges(city_name)
    labels = read_labels(city_name, vertex_id_map)
    graph = Graph().build(edges)
    n_samples = len(labels)
    mp = MeasurePropagationSklearn()
    param_grid = {'mu': [1e-8, 1e-4, 0.01, 0.1, 1, 10, 100],
                  'nu': [1e-8, 1e-6, 1e-4, 0.01, 0.1],
                  'max_iter': [1000],
                  'tol': [1e-1]}

    best_acc, best_params = .0, None
    for params in ParameterGrid(param_grid):
        logging.info("Params: %s" % params)
        mp.set_params(**params)
        accs = []
        for train_index, test_index in StratifiedKFold(n_splits=5).split(np.zeros(n_samples), labels):
            y = labels.copy()
            y[train_index] = -1
            mp.fit(graph, y)
            pred_labels = mp.predict()
            accs.append(accuracy_score(labels[train_index], pred_labels[train_index]))
        acc_mean = np.array(accs).mean()
        logging.info("Accuracy mean %f" % acc_mean)
        if acc_mean > best_acc:
            best_acc = acc_mean
            best_params = params
    logging.info("Best Accuracy mean %f" % best_acc)
    logging.info("Best params %s" % str(best_params))


if __name__ == "__main__":
    main()
