# measure-propagation #

A Python implementation of the paper *Semi-Supervised Learning with Measure Propagation*:
> Subramanya, Amarnag, and Jeff Bilmes. "Semi-supervised learning with measure propagation." Journal of Machine Learning Research 12.Nov (2011): 3311-3370.

This implementation works on Python 3 and follows the [scikit-learn API](https://scikit-learn.org/stable/modules/classes.html).

## Installation
I strongly recommend to use a [virtualenv](https://virtualenv.pypa.io/en/stable/) in order not to break anything of your current enviroment.

Open a terminal and type the following line, it will install the package using pip:

    pip install git+git://github.com/albertbup/measure-propagation.git

## Usage
The following snippet (extracted from **circles.py**) would give you the gist of how to use it:
```python
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

from measure_propagation import similarity
from measure_propagation.graph import KNNGraph
from measure_propagation.wrapper import MeasurePropagationSklearn

# generate ring with inner box
n_samples = 200
X, y = make_circles(n_samples=n_samples, shuffle=False)

# create labels
outer, inner = 0, 1
labels = -np.ones(n_samples, dtype=np.int32)  # -1 is set for unlabeled data points
labels[0] = outer
labels[-1] = inner

# create graph with input data
# here we build a nearest neighbor graph because data do not inherently have a graph structure,
# otherwise you may just import the Graph class and add edges directly (check webkb.py file)
K = 4
graph = KNNGraph(K, similarity.euclidean_similarity).build(X)

# create measure propagation
mp = MeasurePropagationSklearn(mu=0.1,
                               nu=0.01,
                               tol=2e-2,
                               max_iter=100)

# train algorithm
mp.fit(graph, labels)

# evaluate
output_labels = mp.predict()
print("Accuracy %f" % accuracy_score(y, output_labels))
```
