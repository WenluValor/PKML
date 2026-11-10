"""Section 5.2: DNN baseline for the one-dimensional ODE simulation."""
import copy
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import rho


class DNN(nn.Module):
    def __init__(self, d, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BootstrapDNN:
    def __init__(self, model, lr=1e-2, epochs=100,
                 bootstrap_lr=1e-2, bootstrap_epochs=100):
        self.model = model
        self.initial_state = copy.deepcopy(model.state_dict())
        self.lr = lr
        self.epochs = epochs
        self.bootstrap_lr = bootstrap_lr
        self.bootstrap_epochs = bootstrap_epochs

    def _train_model(self, model, X, Y, lr=None, epochs=None):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.lr if lr is None else lr)
        loss_fn = nn.MSELoss()
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        for _ in range(self.epochs if epochs is None else epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(X), Y)
            loss.backward()
            optimizer.step()
        model.eval()
        return model

    def fit(self, X, Y):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        self.model = self._train_model(self.model, self.X, self.Y)
        return self

    def predict(self, X):
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32)).numpy()

    def get_radius(self, alpha=0.05, B=200, L=1, J=1000, seed=0):
        # Same domain and evaluation-grid seed as kernel.fit_band().
        grid_seed, bootstrap_seed = np.random.SeedSequence(seed).spawn(2)
        X_eval = np.random.default_rng(grid_seed).uniform(
            0, L, size=(J, self.X.shape[1]))
        mean_eval = self.predict(X_eval)
        X_tensor = torch.tensor(X_eval, dtype=torch.float32)
        rng = np.random.default_rng(bootstrap_seed)
        maxima = np.empty(B)
        for b in range(B):
            indices = rng.choice(len(self.X), len(self.X), replace=True)
            # Refit each resample from the original initialization, not the fitted weights.
            model_b = copy.deepcopy(self.model)
            model_b.load_state_dict(self.initial_state)
            self._train_model(model_b, self.X[indices], self.Y[indices],
                              lr=self.bootstrap_lr, epochs=self.bootstrap_epochs)
            with torch.no_grad():
                prediction = model_b(X_tensor).numpy()
            maxima[b] = np.max(np.abs(prediction - mean_eval))
        rank = int(np.ceil((1 - alpha) * B - 1e-9)) - 1
        return float(np.sort(maxima)[rank])


def fit_dnn(X, Y, lr, epochs=100, seed=0):
    # Fixed initialization makes learning-rate comparisons reproducible.
    torch.manual_seed(seed)
    model = BootstrapDNN(DNN(X.shape[1]), lr=lr, epochs=epochs)
    return model.fit(X, Y)


def get_lr_epochs(X, Y, lr_grid=None, epochs_grid=None, seed=1,
                  B=200, J=1000, alpha=0.05, fit_fn=None):
    """Return selected (learning_rate, epochs, candidate_results)."""
    if lr_grid is None:
        lr_grid = np.logspace(-5, -1, 9)
    if epochs_grid is None:
        epochs_grid = [50, 100, 300, 1000, 3000]
    if fit_fn is None:
        fit_fn = fit_dnn
    X, Y = np.asarray(X), np.asarray(Y).reshape(-1)
    indices = np.random.default_rng(seed).permutation(len(X))
    training, validation = np.split(indices, [len(X) // 2])
    target = 1 - alpha / len(validation)
    best_key = None
    best_pair = None
    results = []
    for epochs in epochs_grid:
        for lr in lr_grid:
            # Every candidate starts independently from the same initialization.
            model = fit_fn(X[training], Y[training], float(lr), int(epochs), seed)
            radius = model.get_radius(alpha=alpha, B=B, J=J)
            coverage = float(np.mean(
                np.abs(Y[validation] - model.predict(X[validation])) <= radius))
            width = 2 * radius
            key = ((0, width, -coverage) if coverage >= target
                   else (1, -coverage, width))
            results.append(dict(lr=float(lr), epochs=int(epochs),
                                validation_coverage=coverage,
                                bonferroni_target=target, radius=radius, width=width))
            if best_key is None or key < best_key:
                best_key = key
                best_pair = (float(lr), int(epochs))
    return *best_pair, results


def get_lr(X, Y, lr_grid=None, seed=1, epochs=100, B=200, J=1000):
    """Compatibility wrapper for tuning only learning rate at a fixed epoch count."""
    lr, _, results = get_lr_epochs(
        X, Y, lr_grid=lr_grid, epochs_grid=[epochs], seed=seed, B=B, J=J)
    return lr, results


def tune_hyperparameters(n, lr_grid=None, epochs_grid=None, seed=1, B=200, J=1000):
    """Pool the two independent n//4 datasets, then split for joint tuning."""
    p_true = np.array([2 * np.pi, 1.0])  # [omega, beta]
    X_p, Y_p = rho.generate_data(int(n / 4), p_true, d=1, seed=4321)
    X_tune, Y_tune = rho.generate_data(int(n / 4), p_true, d=1, seed=1234)
    return get_lr_epochs(
        np.concatenate((X_p, X_tune)), np.concatenate((Y_p, Y_tune)),
        lr_grid=lr_grid, epochs_grid=epochs_grid, seed=seed, B=B, J=J)


def get_res(X_train, Y_train, test_size, seed=0, test_unit=50,
            lr=1e-2, epochs=100, B=200, J=1000):
    model = fit_dnn(X_train, Y_train, lr, epochs, seed)
    radius = model.get_radius(B=B, J=J)
    X_test, Y_test = rho.generate_data(
        test_unit * test_size, np.array([2 * np.pi, 1.0]),
        d=X_train.shape[1], seed=2026, noise=False)
    prediction = model.predict(X_test)
    covered = np.abs(Y_test - prediction) <= radius
    point_coverage = float(np.mean(covered))
    uniform_coverage = float(covered.reshape(test_size, test_unit).all(axis=1).mean())
    mse = float(np.mean((Y_test - prediction) ** 2))
    return radius, point_coverage, uniform_coverage, mse


def main(ns=(8000, 9000, 10000, 11000, 12000), repetitions=200,
         lr_grid=None, epochs_grid=None, B=200, J=1000, output_dir=None):
    torch.set_num_threads(1)
    p_true = np.array([2 * np.pi, 1.0])  # [omega, beta]
    output_dir = (Path(output_dir) if output_dir is not None
                  else Path(__file__).resolve().parent / "DNN")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for n in ns:
        lr, epochs, tuning = tune_hyperparameters(
            n, lr_grid=lr_grid, epochs_grid=epochs_grid, seed=1, B=B, J=J)
        with (output_dir / f"n{n}_hyperparameter_tuning.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(tuning[0]))
            writer.writeheader()
            writer.writerows(tuning)
        selected.append(dict(n=n, lr=lr, epochs=epochs))
        with (output_dir / "selected_hyperparameters.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["n", "lr", "epochs"])
            writer.writeheader()
            writer.writerows(selected)

        output_path = output_dir / f"n{n}_{repetitions}_repetitions.csv"
        with output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["point_coverage", "uniform_coverage", "MSE", "radius"])
            for seed in range(repetitions):
                # Exactly the same inference data as kernel.main(), row by row.
                X_train, Y_train = rho.generate_data(n, p_true, d=1, seed=seed)
                radius, point, uniform, mse = get_res(
                    X_train, Y_train, test_size=100, seed=seed, lr=lr,
                    epochs=epochs, B=B, J=J)
                writer.writerow([point, uniform, mse, radius])
                file.flush()
        print(f"n={n}, lr={lr:g}, epochs={epochs}; saved {output_path}")


if __name__ == "__main__":
    main()
