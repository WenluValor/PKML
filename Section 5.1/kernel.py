import numpy as np
from itertools import product
import rho

def make_k_grid(Kmax, d):
    """Generate multi-index grid k ∈ Z^d with |k_i| ≤ Kmax"""
    return list(product(range(-Kmax, Kmax + 1), repeat=d))


def phi(k, x, L):
    """Fourier basis φ_k(x)"""
    k = np.array(k)
    x = np.array(x)
    d = x.shape[-1]
    return (4 * L) ** (-d / 2) * np.exp(1j * np.pi / (2 * L) * np.dot(x, k))


def P_vec(Z, coeffs):
    # Z: (M, d), the frequencies xi_k = i*pi*k/(2L)
    # coeffs: dict {alpha_tuple: value}
    # returns P_n(xi_k) = sum_alpha p_alpha xi_k^alpha

    M, d = Z.shape
    result = np.zeros(M, dtype=complex)

    for alpha, c in coeffs.items():
        alpha = np.array(alpha)  # (d,)
        term = np.prod(Z ** alpha, axis=1)  # (M,)
        result += c * term

    return result


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


def F_Omega(r, L=1.0, omega=None):
    """
    (4L)^{-d} * int_Omega exp(i*pi/(2L) <r, x>) dx  for a box Omega = prod_r [a_r, b_r].
    r: (..., d) integer differences k - j
    omega: (d, 2) array of [a_r, b_r]; default [0, L]^d (Section 5: Unif(0,1)^3, L = 1)
    returns: (...) complex
    """
    r = np.asarray(r, dtype=float)
    d = r.shape[-1]
    if omega is None:
        omega = np.array([[0.0, L]] * d)
    omega = np.asarray(omega, dtype=float)
    a, b = omega[:, 0], omega[:, 1]

    t = np.pi * r / (2 * L)                      # (..., d)
    t_safe = np.where(t == 0, 1.0, t)
    # int_a^b exp(i t x) dx = (exp(i t b) - exp(i t a)) / (i t);  = b - a when t = 0
    term = np.where(t == 0, b - a,
                    (np.exp(1j * t * b) - np.exp(1j * t * a)) / (1j * t_safe))

    return np.prod(term, axis=-1) / (4 * L) ** d


def build_G(K_set, coeffs, d, s, gamma, L=1.0, omega=None):
    """
    Build G from eq:op-G:
    G[j, k] = gamma * w_j * 1{j=k}
              + (4L)^{-d} P(xi_k) conj(P(xi_j)) int_Omega exp(i pi/(2L) <k - j, x>) dx
    """
    K_set = np.array(K_set)
    Z = 1j * np.pi * K_set / (2 * L)  # (M, d), xi_k = i*pi*k/(2L)
    P_vals = P_vec(Z, coeffs)
    P_conj = np.conj(P_vals)
    # rows j carry conj(P(xi_j)), columns k carry P(xi_k)
    outer = P_conj[:, None] * P_vals[None, :]
    # diff[j, k] = k - j
    diff = K_set[None, :, :] - K_set[:, None, :]  # (M, M, d)
    F_mat = F_Omega(diff, L=L, omega=omega)
    G = outer * F_mat
    # Sobolev part: gamma * w_j on the diagonal
    G[np.diag_indices_from(G)] += gamma * sobolev_weights(K_set, s, L)

    return G

def K(x, y, coeffs, s, gamma, Kmax=3, L=1.0, omega=None):
    """
    Kernel from eq:K(x,y):
    K(x, y) = sum_k (G^{-1} conj(Phi(y)))_k phi_k(x)
    """
    d = x.shape[-1]
    K_set = make_k_grid(Kmax, d)

    # Build G (pass L through)
    G = build_G(K_set, coeffs, d, s, gamma, L=L, omega=omega)

    # Phi: (M, n_x) and (M, n_y)
    Phi_x = np.array([phi(k, x, L) for k in K_set])
    Phi_y = np.array([phi(k, y, L) for k in K_set])

    # Solve G a = conj(Phi(y))
    a = np.linalg.solve(G, np.conj(Phi_y))

    # K(x, y) = Phi(x)^T a  (no conjugate on Phi(x)); imaginary part is round-off
    return np.real(Phi_x.T @ a)

class PDEKernelRidge:
    def __init__(self, kernel, coeffs, s, gamma, Kmax=3, L=1.0, lam=1e-3):
        self.kernel = kernel
        self.lam = lam
        self.coeffs = coeffs
        self.s = s
        self.gamma = gamma
        self.Kmax = Kmax
        self.L = L

    def _basis(self, X):
        X = np.asarray(X, dtype=np.float64)
        phase = np.pi / (2 * self.L) * (X @ self.positive_frequencies.T)
        basis = np.empty((len(X), len(self.frequencies)))
        basis[:, 0] = 1.0
        basis[:, 1::2] = np.sqrt(2.0) * np.cos(phase)
        basis[:, 2::2] = np.sqrt(2.0) * np.sin(phase)
        return basis * (4 * self.L) ** (-X.shape[1] / 2)

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float64)
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
        G = build_G(self.frequencies, self.coeffs, d, self.s, self.gamma, L=self.L)
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

    def bootstrap_maxima(self, X_eval, B=500, seed=0):
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

    def fit_band(self, alpha=0.05, J=1000, B=500, seed=0, bounds=None):
        # J is the number of evaluation points. Section 5 uses [0,1]^3.
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

def get_lam_gam(X, Y, coeffs, s, seed=0):
    n = X.shape[0]
    lam_grid = np.unique(np.concatenate([
        np.logspace(-14, -1, 14),
        3 * np.logspace(-12, -5, 8)
    ]))
    gam_grid = np.logspace(-10, 0, 11)
    best_key = None
    best_pair = None

    for i, lam in enumerate(lam_grid):
        for j, gamma in enumerate(gam_grid):
            model = PDEKernelRidge(kernel=K, lam=lam, coeffs=coeffs, s=s, gamma=gamma, Kmax=5, L=1.0)
            # Section 5: for each (lambda, gamma), randomly split into two equal halves;
            # the first half fits, the second half evaluates response coverage.
            N = int(n / 2)
            perm = np.random.default_rng([seed, i, j]).permutation(n)
            X_train, Y_train = X[perm[:N]], Y[perm[:N]]
            X_test, Y_test = X[perm[N:]], Y[perm[N:]]

            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            model.fit_band(alpha=0.05, J=1000, B=500)
            radius = model.conf_width(X_test)
            lower = Y_pred - radius
            upper = Y_pred + radius
            covered = (Y_test >= lower) & (Y_test <= upper)
            coverage_rate = np.mean(covered)
            width = np.max(model.eval_width)
            if coverage_rate >= 1 - 0.05 / N:
                key = (0, width, -coverage_rate)
            else:
                key = (1, -coverage_rate, width)
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (lam, gamma)

    return best_pair


def get_res(X_train, Y_train, test_size, lam, gamma, s, coeffs, test_unit=50):
    model = PDEKernelRidge(kernel=K, lam=lam, coeffs=coeffs,
                           s=s, gamma=gamma, Kmax=5, L=1.0)
    model.fit(X_train, Y_train)
    model.fit_band(alpha=0.05, J=1000, B=200)
    d = X_train.shape[1]

    p_true = np.array([0.6, 0.8])
    X_test, Y_test = rho.generate_data(n=test_unit * test_size, p_true=p_true, d=d, seed=2026, noise=False)

    mean = model.predict(X_test)
    radius = model.conf_width(X_test)
    lower = mean - radius
    upper = mean + radius
    res_cov = ((Y_test >= lower) & (Y_test <= upper))
    point_coverage = np.mean(res_cov)
    MSE = np.mean((Y_test - mean) ** 2)

    tmp = np.zeros([test_size])
    for b in range(test_size):
        tmp[b] = np.mean(res_cov[b * test_unit: (b + 1) * test_unit])
    unif_coverage = np.mean(tmp == 1)

    # Preserve the scalar summary returned to exe.py: maximal grid half-width.
    return np.max(model.eval_width), point_coverage, unif_coverage, MSE

def single_test():
    s = 2
    d = 3
    n = 4000
    p_true = np.array([0.6, 0.8])

    hyper_p = {
        4000: [[0.51351212, 0.7880484], 3e-07, 1.0],
        6000: [[0.59286939, 0.638162], 3e-10, 1e-06],
        8000: [[0.53167835, 0.85294609], 1e-14, 1e-10],
        10000: [[0.54790294, 0.73133648], 3e-12, 1e-10],
        12000: [[0.51878106, 0.78550927], 1e-14, 1e-10]
    }
    p, lam, gamma = hyper_p[n]
    coeffs = {
        (2, 0, 0): p[0],
        (0, 2, 0): p[1],
        (0, 0, 2): 1.0,
    }
    # X, Y = rho.generate_data(int(n / 4), p_true, d, seed=1234)
    # lam, gamma = get_lam_gam(X, Y, coeffs, s, seed=1)

    # seed changes with repetitions!
    X, Y = rho.generate_data(n, p_true, d, seed=1)
    model = PDEKernelRidge(kernel=K, lam=lam, coeffs=coeffs,
                           s=s, gamma=gamma, Kmax=5, L=1.0)
    model.fit(X, Y)
    model.fit_band(alpha=0.05, J=1000, B=200)

    test_size = 100
    res_cov = np.zeros([test_size])
    res_MSE = np.zeros([test_size])
    for b in range(test_size):
        X_test, Y_test = rho.generate_data(n=50, p_true=p_true, d=d, seed=b + 2000, noise=False)
        mean = model.predict(X_test)
        radius = model.conf_width(X_test)
        lower = mean - radius
        upper = mean + radius
        res_cov[b] = np.mean((Y_test >= lower) & (Y_test <= upper))
        res_MSE[b] = np.mean((Y_test - mean) ** 2)

    point_coverage = np.mean(res_cov)
    unif_coverage = np.mean(res_cov == 1)
    MSE = np.mean(res_MSE)
    print(point_coverage, unif_coverage)
    print(np.max(model.eval_width))
    print(MSE)

def main(n=9000, repetitions=200):
    import csv
    from pathlib import Path
    s = 2
    d = 3
    p_true = np.array([0.6, 0.8])

    hyper_p = {
        8000: [[0.53167835, 0.85294609], 1e-11, 1e-9],
        9000: [[0.6490463904496712, 0.5894334891534504], 1e-11, 1e-9],
        10000: [[0.54790294, 0.73133648], 3e-12, 1e-10],
        11000: [[0.548540481295538, 0.5901406456873586], 1e-12, 1e-10],
        12000: [[0.51878106, 0.78550927], 3e-12, 1e-10],
    }
    p, lam, gamma = hyper_p[n]
    coeffs = {
        (2, 0, 0): p[0],
        (0, 2, 0): p[1],
        (0, 0, 2): 1.0,
    }
    # X, Y = rho.generate_data(int(n / 4), p_true, d, seed=1234)
    # lam, gamma = get_lam_gam(X, Y, coeffs, s, seed=1)

    output_dir = Path(__file__).resolve().parent / "kernel"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"n{n}_{repetitions}_repetitions.csv"
    test_size = 100
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["point_coverage", "uniform_coverage", "MSE", "radius"])
        for b in range(repetitions):
            X_train, Y_train = rho.generate_data(n, p_true, d, seed=b)
            radius, point_coverage, unif_coverage, MSE = get_res(
                X_train, Y_train, test_size, lam, gamma, s, coeffs)
            writer.writerow([point_coverage, unif_coverage, MSE, radius])
            file.flush()
    print(f"Saved {output_path}")
    return output_path


if __name__ == "__main__":
    main()
