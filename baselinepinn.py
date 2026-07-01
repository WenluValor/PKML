import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rho
import pandas as pd

class DNN(nn.Module):
    def __init__(self, d, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def pde_residual(model, x, p):
    """
    x: (N, d)
    """
    x.requires_grad_(True)

    u = model(x)

    grads = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # Laplacian over first d-1 dims
    lap = 0
    for i in range(x.shape[1]):
        grad_i = grads[:, i]
        grad2_i = torch.autograd.grad(
            grad_i, x,
            grad_outputs=torch.ones_like(grad_i),
            create_graph=True
        )[0][:, i]
        if i == x.shape[1] - 1:
            lap += grad2_i
        else:
            lap += p[i] * grad2_i

    return lap

class BootstrapDNN:
    def __init__(self, model, coeffs, lr=1e-2, epochs=100):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.coeffs = coeffs

    def _train_model(self, model, X, Y, lambda_pde=0.01):
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X)
            loss_data = loss_fn(pred, Y)
            res = pde_residual(self.model, X, self.coeffs)
            loss_pde = torch.mean(res ** 2)
            loss = loss_data + lambda_pde * loss_pde
            loss.backward()
            optimizer.step()

        return model

    def fit(self, X, Y):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)

        self.model = self._train_model(
            self.model,
            self.X,
            self.Y
        )

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            return self.model(X).numpy()

    def get_radius(self, alpha=0.05, B=200, L=1, M=1000):
        np.random.seed(1)
        d = self.X.shape[1]

        X_tilde = np.random.uniform(-L, L, size=(M, d))
        mean_tilde = self.predict(X_tilde)
        n = len(self.X)
        sup_vals = []

        for b in range(B):
            idx = np.random.choice(n, n, replace=True)
            X_boot = self.X[idx]
            Y_boot = self.Y[idx]

            model_b = DNN(d)
            model_b = self._train_model(model_b, X_boot, Y_boot)

            with torch.no_grad():
                pred_b = model_b(torch.tensor(X_tilde, dtype=torch.float32)).numpy()

            sup_vals.append(np.max(np.abs(pred_b - mean_tilde)))

        q = np.quantile(sup_vals, 1 - alpha)

        return q


def get_res(X_train, Y_train, test_size, coeffs, seed, case, test_unit=50):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    d = X_train.shape[1]
    model = DNN(d)
    cp_model = BootstrapDNN(model, coeffs)
    cp_model.fit(X_train, Y_train)
    # radius = cp_model.get_radius()
    n = X_train.shape[0]
    df = pd.read_csv(case + '-1/n' + str(n) + '.csv')
    radius = df.loc[seed, 'rad_pinn']

    if case == 'bias':
        p_true = np.array([0.9, 0.8])
        X_test, Y_test = rho.generate_data(n=test_unit * test_size, p_true=p_true, d=d, seed=2026, noise=False)
    elif case == 'appr':
        p_true = np.array([0.5, 0.5])
        q_true = np.array([0.6, 0.4])
        X_test, Y_test = rho.generate_full_data(n=test_unit * test_size, p_true=p_true, q_true=q_true,
                                                d=d, seed=2026, noise=False)

    mean = cp_model.predict(X_test)
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

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    s = 2
    d = 3
    n = 4000
    # p_true = np.array([0.9, 0.8])
    # X, Y = rho.generate_data(n=n, p_true=p_true, d=d)
    p_true = np.array([0.5, 0.5])
    q_true = np.array([0.6, 0.4])
    X, Y = rho.generate_full_data(n, p_true, q_true, d)
    model = DNN(d)
    cp_model = BootstrapDNN(model, p_true)

    # fit
    cp_model.fit(X, Y)
    radius = cp_model.get_radius()

    # test
    test_size = 20
    res_cov = np.zeros([test_size])
    res_MSE = np.zeros([test_size])
    for b in range(test_size):
        # X_test, Y_test = rho.generate_data(n=200, p_true=p_true, d=d, seed=b+2000, noise=False)
        X_test, Y_test = rho.generate_full_data(n=200, p_true=p_true, q_true=q_true,
                                                d=d, seed=b+2000, noise=False)
        mean = cp_model.predict(X_test)
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


