import logging
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

from measure_propagation import similarity
from measure_propagation.graph import KNNGraph
from measure_propagation.wrapper import MeasurePropagationSklearn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# generate ring with inner box
n_samples = 200
X, y = make_circles(n_samples=n_samples, shuffle=False)

# create labels
outer, inner = 0, 1
labels = -np.ones(n_samples, dtype=np.int32)  # -1 is set for unlabeled data points
labels[0] = outer
labels[-1] = inner

# create a nearest neighbor graph
K = 4
graph = KNNGraph(K, similarity.euclidean_similarity).build(X)

# create measure propagation
mp = MeasurePropagationSklearn(0.1,
                               0.01,
                               2e-2,
                               100)

# run algorithm
start = time.time()
mp.fit(graph, labels)
end = time.time()
logging.debug(end - start)
logger.debug(mp.q)
output_labels = mp.predict()
logging.debug("Accuracy %f" % accuracy_score(y, output_labels))

# visualize results
plt.figure(figsize=(8.5, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[labels == outer, 0], X[labels == outer, 1], color='navy',
            marker='s', lw=0, label="outer labeled", s=10)
plt.scatter(X[labels == inner, 0], X[labels == inner, 1], color='c',
            marker='s', lw=0, label='inner labeled', s=10)
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], color='darkorange',
            marker='.', label='unlabeled')
plt.legend(scatterpoints=1, shadow=False, loc='upper right')
plt.title("Raw data (2 classes=outer and inner)")

plt.subplot(1, 2, 2)
output_label_array = np.asarray(output_labels)
outer_numbers = np.where(output_label_array == outer)[0]
inner_numbers = np.where(output_label_array == inner)[0]
plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color='navy',
            marker='s', lw=0, s=10, label="outer learned")
plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color='c',
            marker='s', lw=0, s=10, label="inner learned")
plt.legend(scatterpoints=1, shadow=False, loc='upper right')
plt.title("Labels learned with Measure Propagation (%d-NN)" % K)

plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)

plt.show()
