import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .mp import MeasurePropagation


class MeasurePropagationSklearn(MeasurePropagation, BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        labeled_indices = np.where(y != -1)[0]
        vertices_labels_dct = {idx: y[idx] for idx in labeled_indices}
        return self.optimize(X, vertices_labels_dct)

    def predict(self):
        return self.get_output_labels()
