"""Exercise 4.8: Metropolis-Hastings for Poisson regression with a Gaussian prior,
compared with the Laplace approximation of Exercise 4.7.

Model (Exercise 4.6):  t_n ~ Poisson(V_n),  V_n = exp(w . x_n),  prior w ~ N(0, s2 I).
We only ever evaluate g(w) = p(t | X, w) p(w), which is proportional to the posterior.

Run:  python scripts/ex4_8_poisson_mh.py            (default: first 20 points)
      python scripts/ex4_8_poisson_mh.py --n 60 --steps 100000
"""
import argparse
import numpy as np


def make_data(n=60, seed=7, w_true=(1.0, 0.7)):
    """Synthetic count data: an intercept column and one covariate u in [-1.5, 1.5]."""
    rng = np.random.RandomState(seed)
    u = rng.uniform(-1.5, 1.5, size=n)
    X = np.column_stack([np.ones(n), u])
    t = rng.poisson(np.exp(X @ np.array(w_true)))
    return X, t


def log_g(w, X, t, s2):
    """log of prior times likelihood, dropping the constants (the t_n! terms, normalisers)."""
    z = X @ w
    return np.sum(t * z - np.exp(z)) - w @ w / (2 * s2)


def gradient(w, X, t, s2):
    V = np.exp(X @ w)
    return X.T @ (t - V) - w / s2


def hessian(w, X, t, s2):
    V = np.exp(X @ w)
    return -(X.T * V) @ X - np.eye(X.shape[1]) / s2


def newton_map(X, t, s2, tol=1e-10, max_iter=100):
    """Newton-Raphson, Equation 4.6, started from w = 0."""
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        step = np.linalg.solve(hessian(w, X, t, s2), gradient(w, X, t, s2))
        w = w - step
        if np.linalg.norm(step) < tol:
            break
    return w


def laplace(X, t, s2):
    """Laplace approximation N(w_hat, Sigma), Sigma = -H^{-1} at the mode (Exercise 4.7)."""
    w_hat = newton_map(X, t, s2)
    Sigma = np.linalg.inv(-hessian(w_hat, X, t, s2))
    return w_hat, Sigma


def metropolis_hastings(X, t, s2, n_steps, step, w0, seed=0):
    """Random-walk Metropolis-Hastings with a symmetric Gaussian proposal N(w, step^2 I).

    Because the proposal is symmetric, the acceptance ratio is g(w*)/g(w); the unknown
    normaliser of the posterior cancels, so only log g is needed.
    """
    rng = np.random.RandomState(seed)
    w = np.array(w0, dtype=float)
    lg = log_g(w, X, t, s2)
    chain = np.empty((n_steps, len(w)))
    accepted = 0
    for i in range(n_steps):
        w_new = w + step * rng.randn(len(w))
        lg_new = log_g(w_new, X, t, s2)
        if np.log(rng.rand()) < lg_new - lg:   # accept with probability min(1, ratio)
            w, lg = w_new, lg_new
            accepted += 1
        chain[i] = w                            # a rejected proposal repeats the old sample
    return chain, accepted / n_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of observations used")
    ap.add_argument("--s2", type=float, default=10.0, help="prior variance")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--burn", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    X, t = make_data()
    X, t = X[: a.n], t[: a.n]
    w_hat, Sigma = laplace(X, t, a.s2)
    sd = np.sqrt(np.diag(Sigma))
    # Rule of thumb for random-walk MH in D dimensions: step about 2.4/sqrt(D) times the posterior sd.
    step = 2.4 / np.sqrt(len(w_hat)) * sd.mean()
    chain, acc = metropolis_hastings(X, t, a.s2, a.steps, step, w0=np.zeros(2), seed=a.seed)
    post = chain[a.burn:]

    np.set_printoptions(precision=4, suppress=True)
    print(f"observations: {a.n}, prior variance: {a.s2}, step: {step:.3f}, acceptance rate: {acc:.2f}")
    print("Laplace:  mean (the mode) =", w_hat, " sd =", sd, " corr = %.3f" % (Sigma[0, 1] / (sd[0] * sd[1])))
    print("MCMC:     mean            =", post.mean(0), " sd =", post.std(0, ddof=1),
          " corr = %.3f" % np.corrcoef(post.T)[0, 1])
    for j, name in enumerate(["w1", "w2"]):
        s = post[:, j]
        skew = np.mean(((s - s.mean()) / s.std()) ** 3)
        print(f"{name}: MCMC mean - mode = {s.mean() - w_hat[j]:+.4f},  MCMC skewness = {skew:+.3f}")


if __name__ == "__main__":
    main()
