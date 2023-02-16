from typing import NamedTuple
import numpy as np
import jax


class qX(NamedTuple):
    """prod_{n=1}^{N} N(X_n | mu^{X}_n, Sigma^{X})"""
    mu: np.array  ## N x D
    Sigma: np.array  ## D x D

    def sum_mean_XXT(self):
        N, D = qX.mu.shape
        return np.sum((qX.mu[n] @ qX.mu[n].T) for n in range(N)) + N * qX.Sigma


class qW(NamedTuple):
    mu: np.array  ## K x D
    Sigma: np.array  ## D x D

    def mean_norm_col(self):
        K, D = qW.mu.shape
        mean_W_D = [(sum(qW.mu[k, d]**2 for k in range(K)) + K * qW.Sigma[d, d]) for d in range(D)]
        return np.array(mean_W_D)

    def mean_WTW(self):
        K, D = qW.mu.shape
        WTW = np.sum((qW.mean[k] @ qW.mean[k].T) for k in range(K)) + K * qW.Sigma
        return WTW

class qmu(NamedTuple):
    mu: np.array  ## K
    Sigma: np.array  ## K x K


class qalpha(NamedTuple):
    a: float
    b: np.array

    def mean(self):
        return np.array([0.5 * qalpha.a / qalpha.b[d] for d in range(len(qalpha.b))])


class qtau(NamedTuple):
    c: float
    d: float

    def mean(self):
        return 0.5 * qtau.c / qtau.d


class