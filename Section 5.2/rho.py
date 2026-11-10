"""Section 5.2: DNN pilot and formula (2) estimation of (omega, beta)."""
import numpy as np
import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, d=1, hidden=64):
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


def compute_derivatives(model, X, batch_size=512):
    """Evaluate the one-dimensional pilot and its first two derivatives."""
    values, first, second = [], [], []
    for start in range(0, len(X), batch_size):
        x = torch.tensor(X[start:start + batch_size],
                         dtype=torch.float64, requires_grad=True)
        u = model(x)
        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        ddu = torch.autograd.grad(du.sum(), x)[0]
        values.append(u.detach().numpy().ravel())
        first.append(du.detach().numpy().ravel())
        second.append(ddu.detach().numpy().ravel())
    return np.concatenate(values), np.concatenate(first), np.concatenate(second)


def ode_coefficients(X, theta):
    """Return p1(x), p0(x) in f'' + p1(x) f' + p0(x) f = 0."""
    omega, beta = theta
    x = np.asarray(X).reshape(-1)
    factor = 1 + beta * x
    return -beta / factor, omega ** 2 * factor ** 2


def estimate_coefficients(X, values, first, second, mu=1.0,
                          bounds=((0.0, 10.0), (0.0, 10.0))):
    """Formula (2), penalizing the L2[0,1] norms of the ODE coefficients.

    For a fixed beta, the objective is quadratic in q=omega**2, so q
    is solved analytically. Only the bounded scalar beta search is numerical.
    The fixed coefficient of f'' is 1; its constant penalty can be omitted.
    """
    x = np.asarray(X).reshape(-1)
    omega_bounds, beta_bounds = bounds
    q_min, q_max = np.square(omega_bounds)

    def profile(beta):
        factor = 1 + beta * x
        a = second - beta / factor * first
        c = factor ** 2 * values
        # Exact integrals: ||p1||_2^2 and ||p0||_2^2 = q^2 * integral4.
        integral1 = beta ** 2 / (1 + beta)
        integral4 = 1 + 2 * beta + 2 * beta ** 2 + beta ** 3 + beta ** 4 / 5
        denominator = np.mean(c ** 2) + mu * integral4
        q = (np.clip(-np.mean(a * c) / denominator, q_min, q_max)
             if denominator > 0 else q_min)
        objective = np.mean((a + q * c) ** 2) + mu * (integral1 + q ** 2 * integral4)
        return float(objective), float(q)

    grid = np.linspace(*beta_bounds, 201)
    scores = np.array([profile(beta)[0] for beta in grid])
    candidates = [float(grid[np.argmin(scores)])]
    for i in range(1, len(grid) - 1):
        if scores[i] <= scores[i - 1] and scores[i] <= scores[i + 1]:
            # Golden-section refinement of each bracketed local minimum.
            left, right = grid[i - 1], grid[i + 1]
            ratio = (np.sqrt(5) - 1) / 2
            c, d = right - ratio * (right - left), left + ratio * (right - left)
            fc, fd = profile(c)[0], profile(d)[0]
            for _ in range(50):
                if fc <= fd:
                    right, d, fd = d, c, fc
                    c = right - ratio * (right - left)
                    fc = profile(c)[0]
                else:
                    left, c, fc = c, d, fd
                    d = left + ratio * (right - left)
                    fd = profile(d)[0]
            candidates.append(float((left + right) / 2))
    beta = min(candidates, key=lambda candidate: profile(candidate)[0])
    return np.array([np.sqrt(profile(beta)[1]), beta])


def get_p(X, Y, seed=0, mu=1.0, hidden=64, max_epochs=3000,
          lbfgs_steps=1000, lr=1e-3, bounds=((0.0, 10.0), (0.0, 10.0))):
    """Return [omega_hat, beta_hat], retaining the Section 5.1 get_p API name."""
    torch.manual_seed(seed)
    X = np.asarray(X).reshape(len(X), -1)
    if X.shape[1] != 1:
        raise ValueError("Section 5.2 requires one-dimensional covariates")
    Y = np.asarray(Y).reshape(-1)
    indices = np.random.default_rng(seed).permutation(len(X))
    pilot, coefficient = np.split(indices, [len(X) // 2])
    model = DNN(d=1, hidden=hidden).double()
    train(model, X[pilot], Y[pilot], lr, max_epochs, lbfgs_steps)
    values, first, second = compute_derivatives(model, X[coefficient])
    return estimate_coefficients(X[coefficient], values, first, second, mu, bounds)


def generate_data(n, p_true, d=1, noise_std=0.1, seed=0, noise=True):
    """Section 5.2: p_true=[omega,beta]; default noise standard deviation .1."""
    if d != 1:
        raise ValueError("Section 5.2 requires d=1")
    omega, beta = p_true
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n, 1))
    x = X[:, 0]
    phase = omega * (x + beta * x ** 2 / 2)
    Y = np.sin(phase) + 0.5 * np.cos(phase)
    if noise:
        Y += noise_std * rng.randn(n)
    return X, Y


if __name__ == "__main__":
    torch.set_num_threads(1)
    n = 12000
    seed = 4321
    p_true = np.array([2 * np.pi, 1.0])  # omega, beta
    X, Y = generate_data(int(n / 4), p_true, seed=seed)
    p_hat = get_p(X, Y, seed=seed)
    print("True (omega, beta):", p_true)
    print("Estimated (omega, beta):", p_hat)
