import numpy as np
import rho

class KRR:
    def __init__(self, kernel, lam=1e-3, noise_var=1.0):
        self.kernel = kernel
        self.lam = lam
        self.noise_var = noise_var

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        K = self.kernel(self.X, self.X)
        n = K.shape[0]

        # (K + λI)^(-1)
        self.K_reg = K + self.lam * n * np.eye(n)
        self.K_inv = np.linalg.inv(self.K_reg)
        self.alpha = self.K_inv @ self.y

    def predict(self, X_test):
        X_test = np.asarray(X_test)
        K_star = self.kernel(X_test, self.X)
        return K_star @ self.alpha

    def post_cov(self, X_test):
        X_test = np.asarray(X_test)

        K_star = self.kernel(X_test, self.X)
        K_star_star = self.kernel(X_test, X_test)

        # Covariance of prediction error
        v = K_star @ self.K_inv
        cov = K_star_star - v @ K_star.T

        # numerical stability
        cov += 1e-8 * np.eye(len(X_test))
        return cov

    def get_radius(self, alpha=0.05, L=1, M=1000):
        np.random.seed(1)
        d = self.X.shape[1]
        X_tilde = np.random.uniform(-L, L, size=(M, d))
        cov = self.post_cov(X_tilde)
        diag_x = np.sqrt(np.diag(cov))

        # ---- Monte Carlo sup-norm ----
        samples = np.random.multivariate_normal(
            mean=np.zeros(len(X_tilde)),
            cov=cov,
            size=M
        )

        sup_vals = np.max(np.abs(samples) / diag_x, axis=1)
        r = np.quantile(sup_vals, 1 - alpha)
        return r

def rbf_kernel(X1, X2, lengthscale=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    sqdist = (
        np.sum(X1**2, axis=1, keepdims=True)
        - 2 * X1 @ X2.T
        + np.sum(X2**2, axis=1)
    )
    return np.exp(-sqdist / (2 * lengthscale**2))

def get_res(X_train, Y_train, test_size, case, test_unit=50):
    d = X_train.shape[1]
    model = KRR(kernel=rbf_kernel, lam=0.01)
    model.fit(X_train, Y_train)
    radius = model.get_radius()

    if case == 'bias':
        p_true = np.array([0.9, 0.8])
        X_test, Y_test = rho.generate_data(n=test_unit * test_size, p_true=p_true, d=d, seed=2026, noise=False)
    elif case == 'appr':
        p_true = np.array([0.5, 0.5])
        q_true = np.array([0.6, 0.4])
        X_test, Y_test = rho.generate_full_data(n=test_unit * test_size, p_true=p_true, q_true=q_true,
                                                d=d, seed=2026, noise=False)
    mean = model.predict(X_test)
    lower = mean - radius * np.sqrt(np.diag(model.post_cov(X_test)))
    upper = mean + radius * np.sqrt(np.diag(model.post_cov(X_test)))
    res_cov = ((Y_test >= lower) & (Y_test <= upper))
    point_coverage = np.mean(res_cov)
    MSE = np.mean((Y_test - mean) ** 2)

    tmp = np.zeros([test_size])
    for b in range(test_size):
        tmp[b] = np.mean(res_cov[b * test_unit: (b + 1) * test_unit])
    unif_coverage = np.mean(tmp == 1)

    rad = np.mean(radius * np.sqrt(np.diag(model.post_cov(X_test))))

    return rad, point_coverage, unif_coverage, MSE

if __name__ == "__main__":
    np.random.seed(0)

    s = 2
    d = 3
    n = 4000
    p_true = np.array([0.9, 0.8])
    X, Y = rho.generate_data(n=n, p_true=p_true, d=d)
    model = KRR(kernel=rbf_kernel, lam=1e-5)
    model.fit(X, Y)
    radius = model.get_radius()

    test_size = 20
    res_cov = np.zeros([test_size])
    res_MSE = np.zeros([test_size])
    for b in range(test_size):
        X_test, Y_test = rho.generate_data(n=200, p_true=p_true, d=d, seed=b+2000, noise=False)
        # X_test, Y_test = rho.generate_full_data(n=200, p_true=p_true, q_true=q_true,
                                                # d=d, seed=b+2000, noise=False)
        mean = model.predict(X_test)
        lower = mean - radius * np.sqrt(np.diag(model.post_cov(X_test)))
        upper = mean + radius * np.sqrt(np.diag(model.post_cov(X_test)))
        res_cov[b] = np.mean((Y_test >= lower) & (Y_test <= upper))
        res_MSE[b] = np.mean((Y_test - mean) ** 2)
        print(b)

    point_coverage = np.mean(res_cov)
    unif_coverage = np.mean(res_cov == 1)
    MSE = np.mean(res_MSE)
    print(point_coverage, unif_coverage)
    print(radius)
    print(MSE)






