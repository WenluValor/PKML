import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


############################################
# 1. Neural network f_theta(x)
############################################
class PINN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.k = d - 1

        self.net = nn.Sequential(
            nn.Linear(d + self.k, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.p_raw = nn.Parameter(torch.ones(self.k) * 0.5)

    def forward(self, x):
        # x: (N, d)
        p = self.get_p()
        p_expanded = p.unsqueeze(0).expand(x.shape[0], -1)  # (N, k)

        x_aug = torch.cat([x, p_expanded], dim=1)  # (N, d+k)

        return self.net(x_aug)

    def get_p(self):
        p = F.softplus(self.p_raw) # + 1e-6  # positivity + stability
        return p



############################################
# 2. PDE residual
############################################
def pde_residual(model, x):
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
    p = model.get_p()
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


############################################
# 3. Training
############################################
def train(model, X_data, Y_data,
          lambda_pde,
          tol=1e-6,
          patience=100,
          max_epochs=7000):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    X_data = torch.tensor(X_data, dtype=torch.float32)
    Y_data = torch.tensor(Y_data, dtype=torch.float32).view(-1,1)

    prev_loss = float('inf')
    stable_steps = 0

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        # Data loss
        pred = model(X_data)
        loss_data = torch.mean((pred - Y_data)**2)

        # PDE loss
        res = pde_residual(model, X_data)
        loss_pde = torch.mean(res**2)

        loss = loss_data + lambda_pde * loss_pde
        loss.backward()
        optimizer.step()

        current_loss = loss.item()

        # convergence check
        if abs(prev_loss - current_loss) < tol:
            stable_steps += 1
        else:
            stable_steps = 0

        prev_loss = current_loss

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss {current_loss:.6f}, p={model.get_p().detach().numpy()}")

        # early stopping
        if stable_steps >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Final loss: {current_loss:.6f}")
            print(f"Estimated p: {model.get_p().detach().numpy()}")
            break

    return model


############################################
# Generate data
############################################
def generate_data(n, p_true, d=3, noise_std=0.01, seed=0, K=2, noise=True):
    # X = (x, t)
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=(n, d))

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    Y = np.zeros(len(X))

    # coefficients with decay
    a = np.random.randn(K, K) * 0.0 + 1

    for k in range(1, K + 1):
        for l in range(1, K + 1):
            lam = np.pi * np.sqrt((p_true[0]) * k ** 2 + (p_true[1]) * l ** 2)

            basis_xy = np.sin(k * np.pi * x) * np.sin(l * np.pi * y)
            basis_z = np.sinh(lam * z) / np.sinh(lam)

            Y += a[k - 1, l - 1] * basis_xy * basis_z

    if noise == True:
        Y += noise_std * np.random.randn(n)
    return X, Y

def generate_full_data(n, p_true, q_true, d=3, noise_std=0.01, seed=0, K=2, noise=True):
    # X = (x, t)
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=(n, d))

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    Y = np.zeros(len(X))

    # coefficients with decay
    a = np.random.randn(K, K) * 0.0 + 1

    for k in range(1, K + 1):
        for l in range(1, K + 1):
            lam = np.pi * np.sqrt((p_true[0]) * k ** 2 + (p_true[1]) * l ** 2)

            basis_xy = np.sin(k * np.pi * x) * np.sin(l * np.pi * y)
            basis_z = np.sinh(lam * z) / np.sinh(lam)

            Y += a[k - 1, l - 1] * basis_xy * basis_z

    for k in range(1, K + 1):
        for l in range(1, K + 1):
            lam = np.pi * np.sqrt((q_true[0]) * k ** 2 + (q_true[1]) * l ** 2)

            basis_xy = np.sin(k * np.pi * x) * np.sin(l * np.pi * y)
            basis_z = np.sinh(lam * z) / np.sinh(lam)

            Y += basis_xy * basis_z / n

    if noise == True:
        Y += noise_std * np.random.randn(n)
    return X, Y


def compute_grad_f(model, x):
    x = x.clone().detach().requires_grad_(True)

    u = model(x)

    grad_p = torch.autograd.grad(
        u,
        model.p_raw,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]

    return grad_p  # shape (d-1,)


def compute_grad_Df(model, x):
    x = x.clone().detach().requires_grad_(True)

    u = model(x)

    grads = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    h = []

    for i in range(x.shape[1] - 1):
        grad_i = grads[:, i]
        grad2_i = torch.autograd.grad(
            grad_i, x,
            grad_outputs=torch.ones_like(grad_i),
            create_graph=True
        )[0][:, i]

        h.append(-grad2_i.item())   # minus sign from PDE

    return np.array(h)  # shape (d-1,)


def get_p(X, Y, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    d = X.shape[1]

    model = PINN(d)
    model = train(model, X, Y, lambda_pde=1.0)
    print("Estimated p:", model.get_p().detach().numpy())

    return model.get_p().detach().numpy()


############################################
# 4. Example
############################################
if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    d = 3
    n = 2500
    mu = 1.0

    '''
    p_true = np.array([0.9, 0.8])
    X, Y = generate_data(n, p_true, d)
    model = PINN(d)

    model = train(model, X, Y, mu)

    torch.save({
        'model_state_dict': model.state_dict(),
        'p': model.get_p().detach().numpy()
    }, 'pinn_model.pth')

    checkpoint = torch.load('pinn_model.pth')

    model = PINN(d)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Estimated p:", model.get_p().detach().numpy())
    '''

    # '''
    p_true = np.array([0.5, 0.5])
    q_true = np.array([0.6, 0.4])
    X, Y = generate_full_data(n, p_true, q_true, d)
    model = PINN(d)

    model = train(model, X, Y, mu)

    torch.save({
        'model_state_dict': model.state_dict(),
        'p': model.get_p().detach().numpy()
    }, 'pinn_full_model.pth')

    checkpoint = torch.load('pinn_full_model.pth')

    model = PINN(d)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Estimated p:", model.get_p().detach().numpy())
    # '''
