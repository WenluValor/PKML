import numpy as np
from itertools import product
import rho
import matplotlib.pyplot as plt

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
    # Z: (M, d)
    # coeffs: dict {alpha_tuple: value}

    M, d = Z.shape
    result = np.zeros(M, dtype=complex)

    for alpha, c in coeffs.items():
        alpha = np.array(alpha)  # (d,)
        term = np.prod(Z ** alpha, axis=1)  # (M,)
        result += c * term

    return result


def F_Omega(r):
    """
    Vectorized version.
    r: (..., d)
    returns: (...)  (product over last axis)
    """
    r = np.asarray(r)

    x = np.pi * r
    numerator = np.sin(x / 2)
    denominator = x

    # safe division: when r == 0 → value = 1
    denominator = np.where(denominator == 0, 1.0, denominator)
    term = np.where(r == 0, 1.0, numerator / denominator)

    return np.prod(term, axis=-1)


def build_G(K_set, coeffs, d, s, gamma, L=1.0):
    """
    Build full G matrix (diagonal + convolution)
    """
    K_set = np.array(K_set)
    Z = -1j * np.pi * np.array(K_set) / (2 * L)  # (M, d)
    P_vals = P_vec(Z, coeffs)
    P_conj = np.conj(P_vals)
    # Outer product
    outer = P_vals[:, None] * P_conj[None, :]
    # Pairwise differences
    diff = K_set[None, :, :] - K_set[:, None, :]  # (M, M, d)
    # Convolution kernel
    F_mat = F_Omega(diff)  # must support vectorized input
    # Combine
    G = outer * F_mat
    # Diagonal term
    norm_sq = np.sum(K_set ** 2, axis=1)
    diag = gamma * (1 + (norm_sq / (2 * L) ** d) ** s)

    G[np.diag_indices_from(G)] += diag

    return G

def K(x, y, coeffs, s, gamma, Kmax=3, L=1.0):
    """
    Full kernel using linear system solve
    """
    d = x.shape[-1]
    K_set = make_k_grid(Kmax, d)

    # Build G
    G = build_G(K_set, coeffs, d, s, gamma)

    # Build Phi(x)
    Phi_x = np.array([phi(k, x, L) for k in K_set])
    # print(Phi_x.shape)
    Phi_y = np.array([phi(k, y, L) for k in K_set])

    # Solve G a = Phi(y)
    a = np.linalg.solve(G, Phi_y)
    # print(a.shape)

    return np.real(np.conj(Phi_x.T) @ a)

def kappa(M, coeffs, s, d, gamma, Kmax=3, L=1.0):
    np.random.seed(1)
    x = np.random.uniform(-2 * L, 2 * L, size=(M, d))
    K_mat = K(x, x, coeffs, s, gamma, Kmax, L)
    max_diag = np.max(np.diag(K_mat))
    return max_diag


class PDEKernelRidge:
    def __init__(self, kernel, coeffs, s, gamma, Kmax=3, L=1.0, lam=1e-3):
        self.kernel = kernel
        self.lam = lam
        self.coeffs = coeffs
        self.s = s
        self.gamma = gamma
        self.Kmax = Kmax
        self.L = L

    def fit(self, X, y):
        """
        X: (n, d)
        y: (n,)
        """
        self.X_train = np.asarray(X)
        self.Y_train = np.asarray(y)
        y = np.asarray(y)

        # Compute kernel matrix
        K = self.kernel(self.X_train, self.X_train, self.coeffs, self.s, self.gamma, self.Kmax, self.L)

        # Regularization
        n = K.shape[0]
        K_reg = K + self.lam * n * np.eye(n)

        # Solve (K + nλI) α = y
        self.alpha = np.linalg.solve(K_reg, y)
        self.K_reg = K_reg
        self.K = K

    def predict(self, X_test):
        """
        X_test: (m, d)
        """
        K_test = self.kernel(np.asarray(X_test), self.X_train, self.coeffs, self.s, self.gamma, self.Kmax, self.L)
        return K_test @ self.alpha

    def conf_width(self, alpha, M, B):
        n = self.X_train.shape[0]
        d = self.X_train.shape[1]
        hat_kappa = kappa(M, self.coeffs, self.s, d, self.gamma, self.Kmax, self.L)
        hat_q = self.compute_quantile(alpha, B)
        A_n = n * self.lam ** (1 + d / (2 * self.s))
        radius = hat_kappa * hat_q / np.sqrt(A_n)
        return radius

    def compute_quantile(self, alpha, B):
        norms = self.bootstrap_norms(B)
        return np.quantile(norms, 1 - alpha)

    def bootstrap_norms(self, B):
        n = self.X_train.shape[0]
        d = self.X_train.shape[1]
        A = np.linalg.solve(self.K_reg, np.eye(n))
        residuals = self.Y_train - self.predict(self.X_train)
        D = np.diag(residuals)
        # H = centering matrix
        H = np.eye(n) - np.ones((n, n)) / n

        # K^{1/2} via eigendecomposition
        eigvals, U = np.linalg.eigh(self.K)
        eigvals = np.maximum(eigvals, 1e-10)
        K_half = U @ np.diag(np.sqrt(eigvals)) @ U.T

        # scaling
        scale = np.sqrt(n * self.lam ** (1 + d / (2 * self.s)))
        Xi = np.random.randn(n, B)  # shape: (n, B)

        # Precompute the linear operator
        M = K_half @ A @ D @ H  # shape: (n, n)

        # Apply to all xi's
        V = M @ Xi  # shape: (n, B)

        # Compute norms column-wise
        norms = scale * np.linalg.norm(V, axis=0)  # shape: (B,)

        return norms


def get_lam_gam(X, Y, coeffs, s):
    n = X.shape[0]
    lam_grid = np.logspace(-5, -1, 11)
    gam_grid = np.linspace(0.1, 0.01, 10)
    scores = np.zeros((len(lam_grid), len(gam_grid))) + 1e4

    for i, lam in enumerate(lam_grid):
        for j, gamma in enumerate(gam_grid):
            model = PDEKernelRidge(kernel=K, lam=lam, coeffs=coeffs, s=s, gamma=gamma, Kmax=5, L=1.0)
            N = int(n * 0.50)
            X_test = X[0: N]
            Y_test = Y[0: N]
            X_train = X[N + 1: n]
            Y_train = Y[N + 1: n]

            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            radius = model.conf_width(alpha=0.05, M=1000, B=500)
            lower = Y_pred - radius
            upper = Y_pred + radius
            covered = (Y_test >= lower) & (Y_test <= upper)
            coverage_rate = np.mean(covered)
            if (coverage_rate >= 1 - 0.05 / N):
                scores[i, j] = radius

    idx = np.argmin(scores)
    i_opt, j_opt = np.unravel_index(idx, scores.shape)
    lam = lam_grid[i_opt]
    gamma = gam_grid[j_opt]
    return lam, gamma

def get_res(X_train, Y_train, test_size, lam, gamma, s, coeffs, case, test_unit=50):
    model = PDEKernelRidge(kernel=K, lam=lam, coeffs=coeffs,
                           s=s, gamma=gamma, Kmax=5, L=1.0)
    model.fit(X_train, Y_train)
    radius = model.conf_width(alpha=0.05, M=1000, B=500)
    d = X_train.shape[1]

    if case == 'bias':
        p_true = np.array([0.9, 0.8])
        X_test, Y_test = rho.generate_data(n=test_unit * test_size, p_true=p_true, d=d, seed=2026, noise=False)
    elif case == 'appr':
        p_true = np.array([0.5, 0.5])
        q_true = np.array([0.6, 0.4])
        X_test, Y_test = rho.generate_full_data(n=test_unit * test_size, p_true=p_true, q_true=q_true,
                                                d=d, seed=2026, noise=False)

    mean = model.predict(X_test)
    lower = mean - radius
    upper = mean + radius
    res_cov = ((Y_test >= lower) & (Y_test <= upper))
    point_coverage = np.mean(res_cov)
    MSE = np.mean((Y_test - mean) ** 2)

    tmp = np.zeros([test_size])
    for b in range(test_size):
        tmp[b] = np.mean(res_cov[b * test_unit: (b + 1) * test_unit])
    unif_coverage = np.mean(tmp == 1)

    return radius, point_coverage, unif_coverage, MSE


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    coeffs = {
        (2, 0, 0): 0.912502,
        (0, 2, 0): 0.82996464,
        # (2, 0, 0): 0.5318166,
        # (0, 2, 0): 0.5187776,
        (0, 0, 2): 1.0,
    }
    s = 2
    d = 3
    n = 4000
    # p_true = np.array([0.5, 0.5])
    p_true = np.array([0.9, 0.8])
    q_true = np.array([0.6, 0.4])
    # X, Y = rho.generate_full_data(n, p_true, q_true, d)
    X, Y = rho.generate_data(n, p_true, d)

    lam = 10**(-5)
    gamma = 0.04
    model = PDEKernelRidge(kernel=K, lam=lam, coeffs=coeffs,
                           s=s, gamma=gamma, Kmax=5, L=1.0)
    model.fit(X, Y)
    radius = model.conf_width(alpha=0.05, M=1000, B=500)

    test_size = 100
    res_cov = np.zeros([test_size])
    res_MSE = np.zeros([test_size])
    for b in range(test_size):
        X_test, Y_test = rho.generate_data(n=200, p_true=p_true, d=d, seed=b + 2000, noise=False)
        # X_test, Y_test = rho.generate_full_data(n=200, p_true=p_true, q_true=q_true,
            # d=d, seed=b+2000, noise=False)
        mean = model.predict(X_test)
        lower = mean - radius
        upper = mean + radius
        res_cov[b] = np.mean((Y_test >= lower) & (Y_test <= upper))
        res_MSE[b] = np.mean((Y_test - mean) ** 2)
        print(b)

    point_coverage = np.mean(res_cov)
    unif_coverage = np.mean(res_cov == 1)
    MSE = np.mean(res_MSE)
    print(point_coverage, unif_coverage)
    print(radius)
    print(MSE)











