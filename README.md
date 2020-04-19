# measure-propagation #

A Python implementation of the paper *Semi-Supervised Learning with Measure Propagation*:
> Subramanya, Amarnag, and Jeff Bilmes. "Semi-supervised learning with measure propagation." Journal of Machine Learning Research 12.Nov (2011): 3311-3370.

This implementation works on Python 3 and follows the [scikit-learn API](http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).

## Installation
I strongly recommend to use a [virtualenv](https://virtualenv.pypa.io/en/stable/) in order not to break anything of your current enviroment.

Open a terminal and type the following line, it will install the package using pip:

    pip install git+git://bitbucket.org/albertbup/measure-propagation.git

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

# create a nearest neighbor graph
K = 4
graph = KNNGraph(K, similarity.euclidean_similarity).build(X)

# create measure propagation
mp = MeasurePropagationSklearn(0.1,
                               0.01,
                               2e-2,
                               100)

# run algorithm
mp.fit(graph, labels)
output_labels = mp.predict()
print("Accuracy %f" % accuracy_score(y, output_labels))
```