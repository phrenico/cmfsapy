import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TimeDelayEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, d=3, tau=1):
        """Time delay.
        Embedd the time series with [len(x) - (d - 1) * tau, d] shape.

        :param int d: embedding dimension
        :param int tau: embedding delay
        """
        self.d = d
        self.tau = tau

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        X = self._embedding(x, self.d, self.tau)
        return X

    def fit_transform(self, x, y=None):
        return self.fit(x).transform(x)

    def _embedding(self, x, d, tau):
        """Time delay embedding

        :param numpy.ndarray x: 1D time series
        :param int d: Embedding dimension
        :param int tau: Embedding delay
        :return: Embedded time series with [len(x) - (d - 1) * tau, d] shape
        :rtype: numpy.ndarray
        """
        embedded_length = len(x) - (d - 1) * tau
        X = np.zeros((embedded_length, d))
        for i in range(d):
            X[:, i] = x[i * tau : embedded_length + i * tau]
        return X