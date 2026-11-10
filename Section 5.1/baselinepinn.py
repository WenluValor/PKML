import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import baselinednn
import rho


P_TRUE = np.array([0.6, 0.8])


class DNN(nn.Module):
    def __init__(self, d, hidden=32):
        super().__init__()
        # A second-order PDE needs a smooth activation with nonzero curvature.
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pde_residual(model, x, p):
    x.requires_grad_(True)
    u = model(x)
    gradients = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    residual = 0
    for j in range(x.shape[1]):
        second = torch.autograd.grad(
            gradients[:, j].sum(), x, create_graph=True)[0][:, j]
        weight = float(p[j]) if j < x.shape[1] - 1 else 1.0
        residual = residual + weight * second
    return residual


class BootstrapDNN(baselinednn.BootstrapDNN):
    """Use the same bootstrap as DNN, with a PDE-penalized training loss."""
    def __init__(self, model, coeffs, lr=1e-2, epochs=100, lambda_pde=0.01):
        super().__init__(model, lr=lr, epochs=epochs)
        self.coeffs = np.asarray(coeffs)
        self.lambda_pde = lambda_pde

    def _train_model(self, model, X, Y):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        Y = torch.tensor(Y, dtype=torch.float32)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            X.grad = None
            loss_data = loss_fn(model(X), Y)
            loss_pde = torch.mean(pde_residual(model, X, self.coeffs) ** 2)
            loss = loss_data + self.lambda_pde * loss_pde
            loss.backward()
            optimizer.step()
        model.eval()
        return model


def fit_pinn(X, Y, coeffs, lr=1e-2, epochs=100, seed=0, lambda_pde=0.01):
    torch.manual_seed(seed)
    model = BootstrapDNN(DNN(X.shape[1]), coeffs, lr=lr, epochs=epochs,
                         lambda_pde=lambda_pde)
    return model.fit(X, Y)


def get_lr_epochs(X, Y, coeffs, lr_grid=None, epochs_grid=None, seed=1,
                  B=200, J=1000, alpha=0.05, lambda_pde=0.01):
    """Tune on a half-training/half-validation split, keeping estimated p fixed."""
    def fit_candidate(x, y, lr, epochs, seed):
        return fit_pinn(x, y, coeffs, lr, epochs, seed, lambda_pde)

    return baselinednn.get_lr_epochs(
        X, Y, lr_grid=lr_grid, epochs_grid=epochs_grid, seed=seed,
        B=B, J=J, alpha=alpha, fit_fn=fit_candidate)


def get_lr_epochs_penalty(X, Y, coeffs, lr_grid=None, epochs_grid=None,
                          lambda_pde_grid=None, seed=1, B=200, J=1000, alpha=0.05):
    """Return selected (learning_rate, epochs, PDE_penalty, candidate_results)."""
    if lambda_pde_grid is None:
        lambda_pde_grid = np.logspace(-4, 0, 5)
    results = []
    for lambda_pde in lambda_pde_grid:
        # Reuse the identical split, initialization and bootstrap seeds for all weights.
        _, _, rows = get_lr_epochs(
            X, Y, coeffs, lr_grid=lr_grid, epochs_grid=epochs_grid,
            seed=seed, B=B, J=J, alpha=alpha, lambda_pde=float(lambda_pde))
        results.extend(dict(row, lambda_pde=float(lambda_pde)) for row in rows)

    def selection_key(row):
        coverage, width = row["validation_coverage"], row["width"]
        return ((0, width, -coverage) if coverage >= row["bonferroni_target"]
                else (1, -coverage, width))

    best = min(results, key=selection_key)
    return best["lr"], best["epochs"], best["lambda_pde"], results


def tune_hyperparameters(n, lr_grid=None, epochs_grid=None, seed=1,
                         B=200, J=1000, lambda_pde_grid=None, p_options=None):
    """Return (estimated_p, learning_rate, epochs, PDE_penalty, candidate_results)."""
    X_p, Y_p = rho.generate_data(int(n / 4), P_TRUE, d=3, seed=4321)
    p_hat = rho.get_p(X_p, Y_p, seed=4321, **(p_options or {}))
    # Estimate p once; tune all three parameters on the independent second dataset.
    X_tune, Y_tune = rho.generate_data(int(n / 4), P_TRUE, d=3, seed=1234)
    lr, epochs, lambda_pde, results = get_lr_epochs_penalty(
        X_tune, Y_tune, p_hat, lr_grid=lr_grid, epochs_grid=epochs_grid,
        seed=seed, B=B, J=J, lambda_pde_grid=lambda_pde_grid)
    return p_hat, lr, epochs, lambda_pde, results


def get_res(X_train, Y_train, test_size, coeffs, seed=0, test_unit=50,
            lr=1e-2, epochs=100, B=200, J=1000, lambda_pde=0.01):
    model = fit_pinn(X_train, Y_train, coeffs, lr, epochs, seed, lambda_pde)
    radius = model.get_radius(B=B, J=J)
    X_test, Y_test = rho.generate_data(
        test_unit * test_size, P_TRUE, d=X_train.shape[1], seed=2026, noise=False)
    prediction = model.predict(X_test)
    covered = np.abs(Y_test - prediction) <= radius
    point_coverage = float(covered.mean())
    uniform_coverage = float(covered.reshape(test_size, test_unit).all(axis=1).mean())
    mse = float(np.mean((Y_test - prediction) ** 2))
    return radius, point_coverage, uniform_coverage, mse


def main(ns=(8000, 9000, 10000, 11000, 12000), repetitions=200,
         lr_grid=None, epochs_grid=None, B=200, J=1000, lambda_pde_grid=None,
         output_dir=None, p_options=None):
    torch.set_num_threads(1)
    output_dir = (Path(output_dir) if output_dir is not None
                  else Path(__file__).resolve().parent / "baselinepinn")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for n in ns:
        p_hat, lr, epochs, lambda_pde, tuning = tune_hyperparameters(
            n, lr_grid=lr_grid, epochs_grid=epochs_grid, seed=1, B=B, J=J,
            lambda_pde_grid=lambda_pde_grid, p_options=p_options)
        with (output_dir / f"n{n}_hyperparameter_tuning.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(tuning[0]))
            writer.writeheader()
            writer.writerows(tuning)
        selected.append(dict(n=n, p1=float(p_hat[0]), p2=float(p_hat[1]),
                             lr=lr, epochs=epochs, lambda_pde=lambda_pde))
        with (output_dir / "selected_hyperparameters.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["n", "p1", "p2", "lr", "epochs", "lambda_pde"])
            writer.writeheader()
            writer.writerows(selected)

        output_path = output_dir / f"n{n}_{repetitions}_repetitions.csv"
        with output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["point_coverage", "uniform_coverage", "MSE", "radius"])
            for seed in range(repetitions):
                X, Y = rho.generate_data(n, P_TRUE, d=3, seed=seed)
                radius, point, uniform, mse = get_res(
                    X, Y, test_size=100, coeffs=p_hat, seed=seed, lr=lr,
                    epochs=epochs, B=B, J=J, lambda_pde=lambda_pde)
                writer.writerow([point, uniform, mse, radius])
                file.flush()
        print(f"n={n}, p={p_hat}, lr={lr:g}, epochs={epochs}, "
              f"lambda_pde={lambda_pde:g}; saved {output_path}")


if __name__ == "__main__":
    main()
