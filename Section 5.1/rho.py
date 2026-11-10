"""Section 5.1: fit a DNN pilot, then estimate PDE coefficients by formula (2)."""
import itertools
import numpy as np
import torch
import torch.nn as nn


############################################
# 1. Neural network f_hat(x)
############################################
class DNN(nn.Module):
    def __init__(self, d=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(2.0 * x - 1.0)


def train(model, X_data, Y_data, lr=1e-3, max_epochs=3000, lbfgs_steps=1000):
    x = torch.tensor(X_data, dtype=torch.float64)
    y = torch.tensor(Y_data, dtype=torch.float64).view(-1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(max_epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()

    if lbfgs_steps:
        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=lbfgs_steps,
            line_search_fn="strong_wolfe", tolerance_grad=1e-9,
            tolerance_change=1e-12
        )

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((model(x) - y) ** 2)
            loss.backward()
            return loss

        optimizer.step(closure)

    model.eval()
    return model


############################################
# 2. Second derivatives and formula (2)
############################################
def compute_hessian_diagonal(model, X, batch_size=512):
    derivatives = []
    for start in range(0, len(X), batch_size):
        x = torch.tensor(X[start:start + batch_size],
                         dtype=torch.float64, requires_grad=True)
        u = model(x)
        grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        diagonal = []
        for j in range(3):
            second = torch.autograd.grad(
                grad[:, j].sum(), x, retain_graph=j < 2
            )[0][:, j]
            diagonal.append(second.detach())
        derivatives.append(torch.stack(diagonal, dim=1).numpy())
    return np.concatenate(derivatives)


def estimate_coefficients(hessian, mu=1.0, bounds=(0.0, 10.0)):
    # Minimize mean((p1*f_xx + p2*f_yy + f_zz)^2) + mu*(p1^2+p2^2).
    A, b = hessian[:, :2], -hessian[:, 2]
    m = len(A)
    C = np.vstack((A / np.sqrt(m), np.sqrt(mu) * np.eye(2)))
    target = np.concatenate((b / np.sqrt(m), np.zeros(2)))
    lower, upper = bounds
    best_value = np.inf

    # Enumerate free/lower/upper states to solve the bounded ridge problem.
    for state in itertools.product(("free", "lower", "upper"), repeat=2):
        free = np.array([s == "free" for s in state])
        p = np.array([upper if s == "upper" else lower for s in state], dtype=float)
        if free.any():
            rhs = target - C[:, ~free] @ p[~free]
            p[free] = np.linalg.lstsq(C[:, free], rhs, rcond=None)[0]
        if np.any(p < lower) or np.any(p > upper):
            continue
        value = np.mean((A @ p - b) ** 2) + mu * (p @ p)
        if value < best_value:
            best_p, best_value = p.copy(), value
    return best_p


def get_p(X, Y, seed=0, mu=1.0, hidden=64, max_epochs=3000,
          lbfgs_steps=1000, lr=1e-3, bounds=(0.0, 10.0)):
    torch.manual_seed(seed)
    X = np.asarray(X)
    Y = np.asarray(Y).reshape(-1)
    indices = np.random.default_rng(seed).permutation(len(X))
    pilot, coefficient = np.split(indices, [len(X) // 2])

    model = DNN(X.shape[1], hidden).double()
    train(model, X[pilot], Y[pilot], lr, max_epochs, lbfgs_steps)
    hessian = compute_hessian_diagonal(model, X[coefficient])
    return estimate_coefficients(hessian, mu, bounds)


############################################
# 3. Section 5.1 data
############################################
def generate_data(n, p_true, d=3, noise_std=0.01, seed=0, K=2, noise=True):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n, d))
    x, y, z = X.T
    Y = np.zeros(n)
    a = rng.randn(K, K) * 0.0 + 1.0

    for k in range(1, K + 1):
        for l in range(1, K + 1):
            lam = np.pi * np.sqrt(p_true[0] * k ** 2 + p_true[1] * l ** 2)
            Y += (a[k - 1, l - 1] * np.sin(k * np.pi * x)
                  * np.sin(l * np.pi * y) * np.sinh(lam * z) / np.sinh(lam))
    if noise:
        Y += noise_std * rng.randn(n)
    return X, Y


if __name__ == "__main__":
    torch.set_num_threads(1)
    n = 3500
    seed = 4321
    p_true = np.array([0.6, 0.8])
    X, Y = generate_data(n, p_true, seed=seed)
    p_hat = get_p(X, Y, seed=seed)
    print("True p:", p_true)
    print("Estimated p:", p_hat)
