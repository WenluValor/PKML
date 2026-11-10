"""Section 5.2: variable-coefficient ODE-induced KRR and uniform bands."""
import csv
from itertools import product
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

import rho


THETA_TRUE = np.array([2 * np.pi, 1.0])


def as_points(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("Section 5.2 requires one-dimensional covariates")
    return X


def make_k_grid(Kmax, d=1):
    return list(product(range(-Kmax, Kmax + 1), repeat=d))


def multi_indices(d, s):
    """All multi-indices alpha in N^d with |alpha| <= s."""
    return [a for a in product(range(s + 1), repeat=d) if sum(a) <= s]


def sobolev_weights(K_set, s, L=1.0):
    """
    w_k = sum_{|alpha|<=s} (pi/(2L))^{2|alpha|} prod_r k_r^{2 alpha_r},  0^0 = 1.
    """
    K_set = np.asarray(K_set, dtype=float)
    M, d = K_set.shape
    w = np.zeros(M)
    for alpha in multi_indices(d, s):
        alpha = np.array(alpha)
        # numpy gives 0.0**0 = 1.0, which matches the 0^0 = 1 convention
        w += (np.pi / (2 * L)) ** (2 * alpha.sum()) * np.prod(K_set ** (2 * alpha), axis=1)
    return w


def build_G(K_set, coeffs, d, s, gamma, L=1.0, quadrature_order=256):
    """G[j,k] = integral conjugate(D phi_j) D phi_k + gamma*w_j*delta_jk.

    D f = f'' - beta/(1+beta*x)*f' + omega**2*(1+beta*x)**2*f.
    Integrate on [0,L] using deterministic Gauss-Legendre quadrature.
    coeffs contains [omega,beta], not a constant-coefficient dictionary.
    """
    if d != 1:
        raise ValueError("Section 5.2 requires d=1")
    frequencies = np.asarray(K_set, dtype=float).reshape(-1)
    nodes, weights = leggauss(quadrature_order)
    X = (L * (nodes + 1) / 2)[:, None]
    weights = L * weights / 2
    p1, p0 = rho.ode_coefficients(X, coeffs)
    z = 1j * np.pi * frequencies / (2 * L)
    basis = np.exp(X * z[None, :]) / np.sqrt(4 * L)
    applied_basis = basis * (z[None, :] ** 2 + p1[:, None] * z[None, :] + p0[:, None])
    G = applied_basis.conj().T @ (weights[:, None] * applied_basis)
    G = (G + G.conj().T) / 2
    G[np.diag_indices_from(G)] += gamma * sobolev_weights(
        frequencies[:, None], s, L)
    return G


def K(x, y, coeffs, s, gamma, Kmax=30, L=1.0, quadrature_order=256):
    x, y = as_points(x), as_points(y)
    frequencies = np.asarray(make_k_grid(Kmax, 1))
    G = build_G(frequencies, coeffs, 1, s, gamma, L, quadrature_order)
    phi_x = np.exp(1j * np.pi * x @ frequencies.T / (2 * L)) / np.sqrt(4 * L)
    phi_y = np.exp(1j * np.pi * y @ frequencies.T / (2 * L)) / np.sqrt(4 * L)
    return np.real(phi_x @ np.linalg.solve(G, phi_y.conj().T))


class PDEKernelRidge:
    def __init__(self, kernel, coeffs, s, gamma, Kmax=30, L=1.0, lam=1e-3, quadrature_order=256):
        self.kernel = kernel
        self.lam = lam
        self.coeffs = np.asarray(coeffs, dtype=float).copy()  # [omega, beta]
        self.s = s
        self.gamma = gamma
        self.Kmax = Kmax
        self.L = L
        self.quadrature_order = quadrature_order

    def _basis(self, X):
        X = as_points(X)
        phase = np.pi / (2 * self.L) * (X @ self.positive_frequencies.T)
        basis = np.empty((len(X), len(self.frequencies)))
        basis[:, 0] = 1.0
        basis[:, 1::2] = np.sqrt(2.0) * np.cos(phase)
        basis[:, 2::2] = np.sqrt(2.0) * np.sin(phase)
        return basis * (4 * self.L) ** (-X.shape[1] / 2)

    def fit(self, X, y):
        self.X_train = as_points(X)
        self.Y_train = np.asarray(y, dtype=np.float64).reshape(-1)
        n, d = self.X_train.shape
        self.frequencies = np.asarray(make_k_grid(self.Kmax, d))
        lookup = {tuple(k): i for i, k in enumerate(self.frequencies)}
        positive = [i for i, k in enumerate(self.frequencies) if tuple(k) > (0,) * d]
        negative = [lookup[tuple(-self.frequencies[i])] for i in positive]
        zero = lookup[(0,) * d]
        self.positive_frequencies = self.frequencies[positive]

        # Sobolev scaling: H = diag(w)^(-1/2) G diag(w)^(-1/2).
        # Its Sobolev penalty is gamma*I; this does not change the kernel.
        weights = np.sqrt(sobolev_weights(self.frequencies, self.s, self.L))
        G = build_G(self.frequencies, self.coeffs, d, self.s, self.gamma, L=self.L,
                    quadrature_order=self.quadrature_order)
        G = (G / weights[:, None]) / weights[None, :]
        self.real_weights = np.concatenate(([weights[zero]], np.repeat(weights[positive], 2)))

        # Sparse unitary conversion to the real constant/cosine/sine basis.
        GQ = np.empty_like(G)
        GQ[:, 0] = G[:, zero]
        GQ[:, 1::2] = (G[:, positive] + G[:, negative]) / np.sqrt(2)
        GQ[:, 2::2] = 1j * (G[:, negative] - G[:, positive]) / np.sqrt(2)
        H = np.empty(G.shape, dtype=np.float64)
        H[0] = GQ[zero].real
        H[1::2] = ((GQ[positive] + GQ[negative]) / np.sqrt(2)).real
        H[2::2] = (1j * (GQ[positive] - GQ[negative]) / np.sqrt(2)).real
        self.chol = np.linalg.cholesky((H + H.T) / 2)

        basis = self._basis(self.X_train) / self.real_weights
        features = np.linalg.solve(self.chol, basis.T).T
        U, singular, Vh = np.linalg.svd(features, full_matrices=False)
        self.singular = singular
        self.U = U
        self.V = Vh.T
        regularization = n * self.lam
        denominator = singular**2 + regularization
        self.ridge_scale = singular / denominator
        self.Uy = U.T @ self.Y_train
        # Retain every SVD direction: lambda supplies the regularization.
        self.projection_coefficients = (
            np.linalg.solve(self.chol.T, self.V) / self.real_weights[:, None]
        ) * self.ridge_scale

        # Stable residual formula, especially near interpolation.
        orthogonal_residual = self.Y_train - U @ self.Uy
        self.residuals = orthogonal_residual + U @ ((regularization / denominator) * self.Uy)
        # W(x) = K_star(x) (K+n*lambda*I)^(-1) diag(residuals) H_n.
        self.R = U.T * self.residuals[None, :]
        self.R -= self.R.mean(axis=1, keepdims=True)
        return self

    def _projection(self, X):
        return self._basis(X) @ self.projection_coefficients

    def predict(self, X_test):
        return self._projection(X_test) @ self.Uy

    def centered_weights(self, X):
        return self._projection(X) @ self.R

    def bootstrap_maxima(self, X_eval, B=200, seed=0):
        W = self.centered_weights(X_eval)
        standard_error = np.linalg.norm(W, axis=1)
        rng = np.random.default_rng(seed)
        Xi = rng.standard_normal((len(self.X_train), B))
        # G_star(x) = sqrt(n) W(x) Xi / V_hat(x)
        # and V_hat(x) = sqrt(n) ||W(x)||_2.
        G_star = np.divide(W @ Xi, standard_error[:, None],
                           out=np.zeros((len(X_eval), B)),
                           where=standard_error[:, None] > 0)
        return np.max(np.abs(G_star), axis=0), standard_error

    def fit_band(self, alpha=0.05, J=1000, B=200, seed=0, bounds=None):
        # J is the number of evaluation points; Section 5.2 uses [0,1].
        d = self.X_train.shape[1]
        if bounds is None:
            bounds = np.array([[0.0, self.L]] * d)
        bounds = np.asarray(bounds)
        grid_rng, multiplier_rng = np.random.SeedSequence(seed).spawn(2)
        rng = np.random.default_rng(grid_rng)
        self.X_eval = rng.uniform(bounds[:, 0], bounds[:, 1], size=(J, d))
        maxima, standard_error = self.bootstrap_maxima(
            self.X_eval, B=B, seed=multiplier_rng)
        # Exact inverse empirical CDF in eq:quantile, without interpolation.
        # small tolerance guards against float round-off in (1 - alpha) * B
        rank = int(np.ceil((1 - alpha) * B - 1e-9)) - 1
        self.critical_value = np.sort(maxima)[rank]
        self.eval_width = standard_error * self.critical_value
        return self

    def conf_width(self, X_test):
        # V_hat(x) c_hat / sqrt(n): one half-width for each test point.
        W = self.centered_weights(X_test)
        return np.linalg.norm(W, axis=1) * self.critical_value

    def confidence_band(self, X_test):
        mean = self.predict(X_test)
        radius = self.conf_width(X_test)
        return mean - radius, mean + radius

def get_lam_gam(X, Y, coeffs, s=2, seed=0, lam_grid=None, gam_grid=None,
                Kmax=30, J=1000, B=200, quadrature_order=256):
    """Bonferroni response-coverage selection on the independent tuning sample."""
    X = as_points(X)
    Y = np.asarray(Y).reshape(-1)
    if lam_grid is None:
        lam_grid = np.unique(np.concatenate([
        np.logspace(-14, -1, 14),
        3 * np.logspace(-12, -5, 8)
    ]))
    if gam_grid is None:
        gam_grid = np.logspace(-10, 0, 11)
    best_key, best_pair = None, None
    for i, lam in enumerate(lam_grid):
        for j, gamma in enumerate(gam_grid):
            permutation = np.random.default_rng([seed, i, j]).permutation(len(X))
            train, validation = np.split(permutation, [len(X) // 2])
            model = PDEKernelRidge(K, coeffs, s, gamma, Kmax=Kmax, lam=lam,
                                   quadrature_order=quadrature_order)
            model.fit(X[train], Y[train])
            model.fit_band(alpha=0.05, J=J, B=B)
            prediction = model.predict(X[validation])
            radius = model.conf_width(X[validation])
            coverage = np.mean(np.abs(Y[validation] - prediction) <= radius)
            width = np.max(model.eval_width)
            target = 1 - 0.05 / len(validation)
            key = ((0, width, -coverage) if coverage >= target
                   else (1, -coverage, width))
            if best_key is None or key < best_key:
                best_key, best_pair = key, (float(lam), float(gamma))
    return best_pair


def get_res(X_train, Y_train, test_size, lam, gamma, s, coeffs, test_unit=50,
            Kmax=30, J=1000, B=200, quadrature_order=256):
    model = PDEKernelRidge(K, coeffs, s, gamma, Kmax=Kmax, lam=lam,
                           quadrature_order=quadrature_order)
    model.fit(X_train, Y_train)
    model.fit_band(alpha=0.05, J=J, B=B)
    X_test, Y_test = rho.generate_data(test_unit * test_size, THETA_TRUE,
                                      d=1, seed=2026, noise=False)
    prediction = model.predict(X_test)
    radius = model.conf_width(X_test)
    covered = np.abs(Y_test - prediction) <= radius
    point_coverage = float(covered.mean())
    uniform_coverage = float(covered.reshape(test_size, test_unit).all(axis=1).mean())
    mse = float(np.mean((Y_test - prediction) ** 2))
    return float(np.max(model.eval_width)), point_coverage, uniform_coverage, mse


def main(ns=(8000, 9000, 10000, 11000, 12000), repetitions=200, hyper_p=None,
         lam_grid=None, gam_grid=None, Kmax=30, J=1000, B=200,
         quadrature_order=256, output_dir=None, p_options=None):
    """Estimate theta and tune once per n, then write 200 fresh inference refits.

    Optional hyper_p={n: [[omega_hat,beta_hat],lambda,gamma]} skips estimation
    and tuning for the supplied sizes.
    """
    output_dir = (Path(output_dir) if output_dir is not None
                  else Path(__file__).resolve().parent / "DE-KRR")
    output_dir.mkdir(parents=True, exist_ok=True)
    s = 2
    selected = []
    for n in ns:
        if hyper_p is not None and n in hyper_p:
            theta, lam, gamma = hyper_p[n]
        else:
            X_p, Y_p = rho.generate_data(int(n / 4), THETA_TRUE, seed=4321)
            theta = rho.get_p(X_p, Y_p, seed=4321, **(p_options or {}))
            X_tune, Y_tune = rho.generate_data(int(n / 4), THETA_TRUE, seed=1234)
            lam, gamma = get_lam_gam(
                X_tune, Y_tune, theta, s, seed=1, lam_grid=lam_grid,
                gam_grid=gam_grid, Kmax=Kmax, J=J, B=B,
                quadrature_order=quadrature_order)
        selected.append(dict(n=n, omega=float(theta[0]), beta=float(theta[1]),
                             lam=float(lam), gamma=float(gamma)))
        with (output_dir / "selected_hyperparameters.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["n", "omega", "beta", "lam", "gamma"])
            writer.writeheader()
            writer.writerows(selected)
        output_path = output_dir / f"n{n}_{repetitions}_repetitions.csv"
        with output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["point_coverage", "uniform_coverage", "MSE", "radius"])
            for seed in range(repetitions):
                X, Y = rho.generate_data(n, THETA_TRUE, seed=seed)
                radius, point, uniform, mse = get_res(
                    X, Y, 100, lam, gamma, s, theta, Kmax=Kmax, J=J,
                    B=B, quadrature_order=quadrature_order)
                writer.writerow([point, uniform, mse, radius])
                file.flush()
        print(f"n={n}, omega={theta[0]:.6g}, beta={theta[1]:.6g}, "
              f"lambda={lam:g}, gamma={gamma:g}; saved {output_path}")


if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)
    main()
    # hyper_p = {
    # 8000: [[3.943789714157792, 0.7486846384268412], 1e-11, 1e-4]
    # 9000: [[3.9508377109515247, 0.8502581249642304], 1e-11, 1e-4]
    # 10000: [[4.085823979796718, 0.9177317625752294], 3e-12, 1e-5]
    # 11000: [[4.52909831768399, 0.6177750189909602], 3e-12, 1e-5]
    # 12000: [[5.229096635266735, 0.26944064956632885], 3e-12, 1e-5]
    # }