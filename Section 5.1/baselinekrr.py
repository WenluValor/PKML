"""Baseline KRR (RBF kernel) with Gaussian-process sup-norm confidence bands.

For each n in (8000, 9000, 10000, 11000, 12000):
  1. Tune lambda: fit KRR and build the band on int(n/4) samples (seed=4321),
     test coverage on another int(n/4) samples (seed=1234).  If the coverage
     reaches the Bonferroni target 1 - 0.05/N_val, choose the narrowest band;
     otherwise choose the highest coverage (ties broken by width).
  2. Run 200 repetitions on the inference data rho.generate_data(n, seed=b),
     b = 0, ..., 199, and write one CSV row per repetition.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, solve_triangular, LinAlgError

import rho

P_TRUE = np.array([0.6, 0.8])
ALPHA = 0.05

def rbf_kernel(X1, X2, lengthscale=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    sqdist = (np.sum(X1 ** 2, axis=1, keepdims=True)
              - 2 * X1 @ X2.T
              + np.sum(X2 ** 2, axis=1))
    np.maximum(sqdist, 0.0, out=sqdist)
    return np.exp(-sqdist / (2 * lengthscale ** 2))


def eval_grid(d, J=1000, L=1.0, seed=0):
    """Same evaluation domain [0, L]^d and seeding scheme as kernel.fit_band()."""
    grid_seed, mc_seed = np.random.SeedSequence(seed).spawn(2)
    X_eval = np.random.default_rng(grid_seed).uniform(0, L, size=(J, d))
    return X_eval, np.random.default_rng(mc_seed)


def critical_value(cov, rng, alpha=ALPHA, J=1000):
    """(1-alpha) quantile of sup_x |Z(x)| / sd(x) for Z ~ N(0, cov)."""
    cov = (cov + cov.T) / 2 + 1e-8 * np.eye(len(cov))
    sd = np.sqrt(np.diag(cov))
    samples = rng.multivariate_normal(np.zeros(len(cov)), cov, size=J,
                                      method="eigh")
    sup_vals = np.max(np.abs(samples) / sd, axis=1)
    # Exact inverse empirical CDF, as in kernel.fit_band().
    rank = int(np.ceil((1 - alpha) * J - 1e-9)) - 1
    return float(np.sort(sup_vals)[rank])


class KRR:
    def __init__(self, kernel=rbf_kernel, lam=1e-3):
        self.kernel = kernel
        self.lam = lam

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = len(self.X)
        K = self.kernel(self.X, self.X)
        K[np.diag_indices(n)] += self.lam * n
        try:
            # Cholesky of K + n*lambda*I (in place, no explicit inverse).
            self.chol = cho_factor(K, lower=True, overwrite_a=True,
                                   check_finite=False)[0]
            self.eig = None
            self.alpha = solve_triangular(
                self.chol.T, solve_triangular(self.chol, self.y, lower=True),
                lower=False)
        except LinAlgError:
            # Very small lambda: K + n*lambda*I is numerically singular.
            # Fall back to an eigendecomposition with negative eigenvalues clipped.
            K = self.kernel(self.X, self.X)
            e, Q = np.linalg.eigh(K)
            del K
            e = np.maximum(e, 0.0) + self.lam * n
            self.chol = None
            self.eig = (Q, e)
            self.alpha = Q @ ((Q.T @ self.y) / e)
        return self

    def _whiten(self, K_star):
        """Return V with V^T V = K_star (K + n lam I)^{-1} K_star^T."""
        if self.chol is not None:
            return solve_triangular(self.chol, K_star.T, lower=True,
                                    check_finite=False)
        Q, e = self.eig
        return (Q.T @ K_star.T) / np.sqrt(e)[:, None]

    def predict(self, X_test):
        return self.kernel(np.asarray(X_test), self.X) @ self.alpha

    def predict_with_sd(self, X_test):
        K_star = self.kernel(np.asarray(X_test), self.X)
        mean = K_star @ self.alpha
        V = self._whiten(K_star)
        # RBF: k(x, x) = 1.  Same jitter as before (1e-8).
        var = 1.0 - np.sum(V ** 2, axis=0) + 1e-8
        return mean, np.sqrt(np.maximum(var, 1e-8))

    def post_cov(self, X_test):
        X_test = np.asarray(X_test)
        V = self._whiten(self.kernel(X_test, self.X))
        return self.kernel(X_test, X_test) - V.T @ V

    def get_radius(self, alpha=ALPHA, J=1000, L=1.0, seed=0):
        """Critical value c such that mean +/- c * sd(x) is a uniform band."""
        X_eval, rng = eval_grid(self.X.shape[1], J, L, seed)
        cov = self.post_cov(X_eval)
        c =critical_value(cov, rng, alpha, J)
        sd_eval = np.sqrt(np.maximum(np.diag(cov), 0) + 1e-8)
        self.eval_halfwidth = c * sd_eval
        return c


############################################
# Lambda tuning
############################################
def default_lam_grid():
    # Same lambda grid as kernel.get_lam_gam().
    return np.unique(np.concatenate([np.logspace(-14, -1, 14),
                                     3 * np.logspace(-12, -5, 8)]))


def default_ls_grid():
    # RBF lengthscale grid, tuned jointly with lambda.
    return np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])


def get_lam(X_fit, Y_fit, X_val, Y_val, lam_grid=None, J=1000, ls_grid=None):
    """Choose lambda for the KRR baseline.

    For every lambda: fit KRR and build the uniform band on (X_fit, Y_fit),
    then check its coverage on the separate sample (X_val, Y_val).
    Same criterion as kernel.get_lam_gam(): if coverage >= 1 - 0.05/N_val,
    choose the narrowest band (max half-width over the evaluation grid);
    if no lambda reaches it, choose the highest coverage.
    """
    if lam_grid is None:
        lam_grid = default_lam_grid()
    if ls_grid is None:
        ls_grid = default_ls_grid()
    target = 1 - ALPHA / len(X_val)
    best_key, best_lam, best_ls, results = None, None, None, []
    # Joint grid search over (lambda, lengthscale).
    for ls in ls_grid:
        kernel = lambda A, B, ls=float(ls): rbf_kernel(A, B, ls)
        for lam in lam_grid:
            # Exactly the same fit / band as get_res() uses.
            model = KRR(kernel=kernel, lam=float(lam)).fit(X_fit, Y_fit)
            c = model.get_radius(J=J)
            mean, sd = model.predict_with_sd(X_val)
            coverage = float(np.mean(np.abs(Y_val - mean) <= c * sd))
            width = float(np.max(model.eval_halfwidth))
            key = ((0, width, -coverage) if coverage >= target
                   else (1, -coverage, width))
            results.append(dict(lam=float(lam), lengthscale=float(ls),
                                validation_coverage=coverage,
                                bonferroni_target=target, max_halfwidth=width,
                                critical_value=c))
            if best_key is None or key < best_key:
                best_key, best_lam, best_ls = key, float(lam), float(ls)
    return best_lam, best_ls, results


############################################
# One repetition
############################################
def get_res(X_train, Y_train, test_size, test_unit=50, lam=0.01,
            J=1000, lengthscale=1.0):
    d = X_train.shape[1]
    kernel = lambda A, B: rbf_kernel(A, B, lengthscale)
    model = KRR(kernel=kernel, lam=lam).fit(X_train, Y_train)
    c = model.get_radius(J=J)

    X_test, Y_test = rho.generate_data(n=test_unit * test_size, p_true=P_TRUE,
                                       d=d, seed=2026, noise=False)
    mean, sd = model.predict_with_sd(X_test)
    halfwidth = c * sd
    covered = np.abs(Y_test - mean) <= halfwidth
    point_coverage = float(np.mean(covered))
    unif_coverage = float(covered.reshape(test_size, test_unit).all(axis=1).mean())
    MSE = float(np.mean((Y_test - mean) ** 2))
    rad = float(np.max(model.eval_halfwidth))  # maximum half-width over J evaluation points
    return rad, point_coverage, unif_coverage, MSE


def tune_lambdas(ns=(8000, 9000, 10000, 11000, 12000), lam_grid=None,
                 J=1000, output_dir=None, ls_grid=None):
    """Tune lambda for each n; save per-n tuning tables and selected_lambdas.csv."""
    d = 3
    output_dir = (Path(output_dir) if output_dir is not None
                  else Path(__file__).resolve().parent / "baselinekrr")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {}
    for n in ns:
        # Fit + band on int(n/4) samples (seed=4321);
        # test coverage on another int(n/4) samples (seed=1234).
        X_fit, Y_fit = rho.generate_data(int(n / 4), P_TRUE, d=d, seed=4321)
        X_val, Y_val = rho.generate_data(int(n / 4), P_TRUE, d=d, seed=1234)
        lam, ls, tuning = get_lam(X_fit, Y_fit, X_val, Y_val, lam_grid=lam_grid,
                                  J=J, ls_grid=ls_grid)
        selected[n] = (lam, ls)
        with (output_dir / f"n{n}_lambda_tuning.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(tuning[0]))
            writer.writeheader()
            writer.writerows(tuning)
        with (output_dir / "selected_lambdas.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["n", "lam", "lengthscale"])
            writer.writerows([k, *v] for k, v in selected.items())
        print(f"n={n}: selected lambda={lam:g}, lengthscale={ls:g}", flush=True)
    return selected


def main(ns=(8000, 9000, 10000, 11000, 12000), repetitions=200,
         lam_grid=None, J=1000, output_dir=None, lambdas=None, ls_grid=None):
    """Run the 200 repetitions.  lambdas: {n: (lam, lengthscale)}; tuned first if not given."""
    d = 3
    output_dir = (Path(output_dir) if output_dir is not None
                  else Path(__file__).resolve().parent / "baselinekrr")
    output_dir.mkdir(parents=True, exist_ok=True)
    if lambdas is None:
        lambdas = tune_lambdas(ns, lam_grid=lam_grid, J=J, output_dir=output_dir,
                               ls_grid=ls_grid)
    for n in ns:
        lam, ls = lambdas[n]
        output_path = output_dir / f"n{n}_{repetitions}_repetitions.csv"
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["point_coverage", "uniform_coverage", "MSE", "radius"])
            for seed in range(repetitions):
                # Same inference data as kernel.main(): seed b on round b+1.
                X_train, Y_train = rho.generate_data(n, P_TRUE, d=d, seed=seed)
                rad, point, uniform, mse = get_res(
                    X_train, Y_train, test_size=100, lam=lam, J=J,
                    lengthscale=ls)
                writer.writerow([point, uniform, mse, rad])
                f.flush()
                if seed % 20 == 0:
                    print(f"  n={n}, repetition {seed}", flush=True)
        print(f"n={n} (lambda={lam:g}, lengthscale={ls:g}): saved {output_path}", flush=True)


if __name__ == "__main__":
    # lambda = 3e-7, lengthscale=0.2
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        tune_lambdas()
    else:
        main()
