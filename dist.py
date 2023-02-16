from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.special import digamma, gammaln


class PolyaGamma(NamedTuple):
    "PG(b,c) distribution"
    b: float
    c: float

    def mean(self) -> float:
        b, c = self
        c0 = jnp.isclose(c, 0.0)
        c_safe = jnp.where(c0, 1.0, c)
        return jnp.where(c0, b / 4, b / (2 * c_safe) * jnp.tanh(c_safe / 2))

    def kl(self, other: "PolyaGamma") -> float:
        """d(self || other) for two PG(b,c) distributions with same shape parameter"""
        # 👀 requirement of same shape parameter is not checked! it will just fail.
        # PG is in the exponential family with
        # f(x) = p0(x) * exp(-c^2/2 x - A(c)) where
        # A(c)=log \int p0(x) exp(-c^2/2 x) = log L(c^2/2) where L(t)=E_p0(exp(t w)) is the Laplace transform
        # When t=c^2/2 we have L(t)=[cosh(c/2)]^{-b}
        # \int p(x) log (p(x)/q(x)) = \int p(x) (-(c_p^2 - c_q^2)/2 x + A(c_q) - A(c_p))
        # = -(c_p^2 - c_q^2)/2 * E_p X + A(c_q) - A(c_p)
        def logcosh(x):
            # only evaluate exp(x) for x<0. can underflow but won't overflow.
            s = abs(x)
            p = jnp.exp(-2 * s)
            return s + jnp.log1p(p) - np.log(2)

        def A(c):
            return -self.b * logcosh(c / 2)

        ret = -(self.c**2 - other.c**2) / 2.0 * self.mean() + A(other.c) - A(self.c)

        return jnp.where(jnp.isclose(self.b, other.b), ret, jnp.nan)


class MVNorm(NamedTuple):
    """Multivariate Normal distribution.
    Params:
        mu: [D] mean
        Sigma: [D, D], variance
    """

    mu: jnp.ndarray
    Sigma: jnp.ndarray

    @property
    def n(self):
        return self.mu.shape[-1]

    @property
    def E_X(self):
        return self.mu

    @property
    def E_X2(self):
        return self.var_X + self.E_X**2


    @property
    def E_XXT(self):
        return self.Sigma + jnp.outer(self.mu, self.mu)

    @property
    def var_X(self):
        return jnp.diagonal(self.Sigma, axis1=-2, axis2=-1)

    def mode(self):
        return self.mu

    def product(self, other: "MVNorm"):
        """If self=N(mu1, Sigma1) and other=N(mu1, Sigma2) then return the normal distribution corresponding to the
        product of the two PDFs"""
        assert self.n == other.n
        S = self.Sigma + other.Sigma
        f = lambda A: jnp.linalg.solve(S, A)
        Sigma = self.Sigma @ f(other.Sigma)
        mu = other.Sigma @ f(self.mu) + self.Sigma @ f(other.mu)
        return MVNorm(mu, Sigma)

    def entropy(self) -> float:
        "-E_q log q(x) "
        H = 0.5 * self.n * (1 + jnp.log(2 * jnp.pi)) + 0.5 * jnp.linalg.slogdet(self.Sigma)[1]

        return H


    def kl(self, other: "MVNorm") -> float:
        """d_KL(p_1 || p_2) where p_i ~ N(mu_i, Sigma_i)"""
        p1 = self
        p2 = other

        def logdet(S):
            return jnp.linalg.slogdet(S)[1]

        def Sinv(S, x):
            return jnp.linalg.solve(S, x)

        ld1, ld2 = [logdet(S) for S in (p1.Sigma, p2.Sigma)]
        dmu = p2.mu - p1.mu
        # https://stanford.edu/~jduchi/projects/general_notes.pdf p. 13
        return 0.5 * (
            ld2
            - ld1
            - p1.n
            + jnp.trace(Sinv(p2.Sigma, p1.Sigma))
            + dmu.dot(Sinv(p2.Sigma, dmu))
        )

class Gamma(NamedTuple):
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / self.beta

    @property
    def E_log(self) -> float:
        return digamma(self.alpha) - jnp.log(self.beta)


    def kl(self, other: "Gamma") -> float:
        p1 = self
        p2 = other

        return (p1.alpha * jnp.log(p1.beta) - gammaln(p1.alpha)
                - (p2.alpha * jnp.log(p2.beta) - gammaln(p2.alpha))
                + (p1.alpha - p2.alpha) * (digamma(p1.alpha) - jnp.log(p1.beta))
                - (p1.beta - p2.beta) * p1.alpha / p1.beta
                )



#
# N, D = 100, 5
# X = MVNorm(mu=jnp.zeros(N, D), Sigma=jnp.eye(D))
# jnp.zeros((N, D))