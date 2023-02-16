# from params import *
from dist import *
import numpy as np
import matplotlib.pyplot as plt
import jax
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

def initiate_params(X_init, W_init, mu_init, beta, a, b, c, d, N, L, D):
    q_X_n = [MVNorm(mu=X_init[n], Sigma=jnp.eye(D)) for n in range(N)]
    q_X = MVNorm(mu=jnp.array([q.mu for q in q_X_n]), Sigma=jnp.array([q.Sigma for q in q_X_n]))

    q_mu = MVNorm(mu=mu_init, Sigma=1/beta * jnp.eye(L))

    q_W_l = [MVNorm(mu=W_init[l], Sigma=jnp.eye(D)) for l in range(L)]
    q_W = MVNorm(mu=jnp.array([q.mu for q in q_W_l]), Sigma=jnp.array([q.Sigma for q in q_W_l]))

    q_alpha_d = [Gamma(alpha=a, beta=b) for d in range(D)]
    q_alpha = Gamma(alpha=jnp.array([q.alpha for q in q_alpha_d]), beta=jnp.array([q.beta for q in q_alpha_d]))

    q_tau = Gamma(alpha=c, beta=d)

    return q_X, q_X_n, q_mu, q_W, q_W_l, q_alpha, q_alpha_d, q_tau

def elbo_ppca(R, q_X, q_X_n, q_mu, q_W, q_W_l, q_alpha, q_alpha_d, q_tau, beta_add, ssq_err):
    E_logR = 0.5 * N * L * (q_tau.E_log - jnp.log(2 * jnp.pi)) - 0.5 * ssq_err
    E_logW = 0.5 * L * sum((q_alpha_d[d].E_log - jnp.log(2 * jnp.pi)) for d in range(D)) - 0.5 * sum((q_alpha_d[d].mean * beta_add[d]) for d in range(D))
    H_q_W = 0.5 * jnp.linalg.slogdet(q_W.Sigma[0])[1] + D / 2 * (1 + jnp.log(2 * jnp.pi))
    KL_terms = (
        q_tau.kl(Gamma(alpha=1., beta=1.))
        + sum(q_alpha_d[d].kl(Gamma(alpha=1., beta=1.)) for d in range(D))
        + q_mu.kl(MVNorm(mu=jnp.zeros(L), Sigma=1/beta * jnp.eye(L)))
        + sum(q_X_n[n].kl(MVNorm(mu=jnp.zeros(D), Sigma=jnp.eye(D))) for n in range(N))
    )
    elbo = E_logR + E_logW + H_q_W - KL_terms

    return elbo



D_true= 2
L = 10
N = 2000
R_lowdim = np.vstack((multivariate_normal.rvs(mean=jnp.ones(D_true) * 5, cov=jnp.diag(jnp.array([8., 2.])), size=int(N/2)),
                     multivariate_normal.rvs(mean=-jnp.ones(D_true) * 5, cov=jnp.diag(jnp.array([4., 4.])), size=int(N/2))))

W_true = multivariate_normal.rvs(mean=jnp.zeros(L), cov=jnp.eye(L) * 10, size=D_true) + 3.
mu_true = jnp.ones(L)
plt.scatter(R_lowdim[:, 0], R_lowdim[:, 1])
plt.show()

R = R_lowdim @ W_true + mu_true + multivariate_normal.rvs(mean=jnp.zeros(L), cov=jnp.ones(L), size=N)

### variational PCA
mu_init = R.mean(0)
U, S, VT = np.linalg.svd(R - R.mean(0))
D = 3
# W_init = VT[:2].T   ## D orthogonal L-dimensional vectors
W_init = multivariate_normal.rvs(mean=jnp.ones(D) * 5, cov=jnp.eye(D), size=L)
X_init = U[:, :D] @ np.diag(S[:D])
beta = 10.
a, b, c, d = 1., 1., 1., 1.

q_X, q_X_n, q_mu, q_W, q_W_l, q_alpha, q_alpha_d, q_tau = initiate_params(X_init, W_init, mu_init, beta=beta,
                                                 a=a, b=b, c=c, d=d, N=N, L=L, D=D)
plt.scatter(q_X.mu[:, 0], q_X.mu[:, 1])
plt.show()

iter_num = 20
for _ in range(iter_num):
    q_X_Sigma_new = jnp.linalg.inv(np.eye(D) + q_tau.mean * sum(q_W_l[l].E_XXT for l in range(L)))
    q_X_mu_new = [q_tau.mean * q_X_Sigma_new @ (q_W.mu.T @ (R[n] - q_mu.mu)) for n in range(N)]
    q_X_n = [MVNorm(mu=q_X_mu_new[n], Sigma=q_X_Sigma_new) for n in range(N)]
    q_X = MVNorm(mu=jnp.array([q.mu for q in q_X_n]), Sigma=jnp.array([q.Sigma for q in q_X_n]))

    q_mu_Sigma_new = 1 / (beta + N * q_tau.mean) * jnp.eye(L)
    q_mu_mu_new = q_tau.mean / (beta + N * q_tau.mean) * sum((R[n] - q_W.mu @ q_X_n[n].mu) for n in range(N))
    q_mu = MVNorm(mu=q_mu_mu_new, Sigma=q_mu_Sigma_new)

    q_W_Sigma_new = jnp.linalg.inv(jnp.diag(q_alpha.mean) + q_tau.mean * sum(q_X_n[n].E_XXT for n in range(N)))
    q_W_mu_new = [(q_tau.mean * q_W_Sigma_new @ sum(((R[n, l] - q_mu.mu[l]) * q_X.mu[n]) for n in range(N)))
                  for l in range(L)]
    q_W_l = [MVNorm(mu=q_W_mu_new[l], Sigma=q_W_Sigma_new) for l in range(L)]
    q_W = MVNorm(mu=jnp.array([q.mu for q in q_W_l]), Sigma=jnp.array([q.Sigma for q in q_W_l]))

    beta_add = [((q_W.mu.sum(0)[d])**2 + L * q_W_Sigma_new[d, d]) for d in range(D)]
    q_alpha_d = [Gamma(alpha=a + 0.5 * L, beta=b + beta_add[d] / 2) for d in range(D)]
    q_alpha = Gamma(alpha=jnp.array([q.alpha for q in q_alpha_d]), beta=jnp.array([q.beta for q in q_alpha_d]))


    E_WTW = sum(q_W_l[l].E_XXT for l in range(L))
    ssq_err = sum((jnp.linalg.norm(R[n])**2 + jnp.linalg.norm(q_mu.mu)**2 + jnp.trace(q_mu.Sigma) + jnp.trace(E_WTW @ q_X_n[n].E_XXT)
                       + 2 * q_mu.mu @ q_W.mu @ q_X.mu[n] - 2 * R[n] @ q_W.mu @ q_X.mu[n] - 2 * R[n] @ q_mu.mu) for n in
                      range(N))
    # print(ssq_err)
    q_tau = Gamma(alpha=c + N * L / 2, beta=d + 0.5 * ssq_err)
    print(elbo_ppca(R, q_X, q_X_n, q_mu, q_W, q_W_l, q_alpha, q_alpha_d, q_tau, beta_add, ssq_err))
plt.scatter(q_X.mu[:, 1], q_X.mu[:, 2])
plt.show()





