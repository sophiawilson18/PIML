import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import tqdm


# Helper Functions
def oscillator(d, w0, x):
    """Analytical solution to the 1D underdamped harmonic oscillator."""
    assert d < w0, "Damping coefficient must be less than angular frequency for underdamped motion."
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d / w)
    A = 1 / (2 * np.cos(phi))
    y = torch.exp(-d * x) * 2 * A * torch.cos(phi + w * x)
    return y


def plot_result(x, y, x_data, y_data, yh, xp=None, step=None):
    """Plot training results with optional physics loss locations."""
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.plot(x, yh, color="tab:blue", linewidth=4, alpha=0.8, label="NN prediction")
    plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.4, label="Training data")
    if xp is not None:
        plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green", alpha=0.4,
                    label="Physics loss training locations")
    plt.legend(loc="upper right", fontsize="large", frameon=False)
    if step is not None:
        plt.text(0.9, 0.8, f"Training step: {step}", fontsize="large", transform=plt.gca().transAxes)
    plt.ylim(-1.1, 1.1)
    plt.axis("off")


# Custom Classes
class SnakeActivation(nn.Module):
    def forward(self, x):
        return x + torch.sin(x) ** 2


class MLP(nn.Module):
    """Multi-Layer Perceptron."""
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(SnakeActivation())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Main Workflow
if __name__ == "__main__":
    # Parameters
    d, w0 = 2, 30
    N_total = 30000
    N_train = int(N_total * 0.5)
    hidden_sizes = [64, 64]
    num_epochs = 100000
    batch_size = 25
    save_path = "plots"
    os.makedirs(save_path, exist_ok=True)
    criterion = nn.MSELoss()

    # Generate Data
    x = torch.linspace(0, 1, N_total).view(-1, 1)
    y = oscillator(d, w0, x).view(-1, 1)

    # Train-Test Split
    x_train = x[:N_train]
    y_train = y[:N_train]
    x_test = x[N_train:]
    y_test = y[N_train:]

    # Data Loader
    train_dataloader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    # Plot Initial Data
    plt.figure()
    plt.plot(x, y, label="Exact solution")
    plt.scatter(x_train, y_train, color="tab:orange", label="Training data")
    plt.legend()
    plt.savefig("initial_data_nn.png")
    plt.close()

    # Define Model and Optimizer
    model = MLP(input_size=1, hidden_sizes=hidden_sizes, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Training Loop
    with tqdm.tqdm(total=num_epochs) as pbar:
        for step in range(num_epochs):
            # Training phase
            model.train()

            for inputs, targets in train_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pbar.update(1)
        
    # Prediction
    with torch.no_grad():
        model.eval()
        y_pred = model(x_test).detach()
    
    # Plot Results
    plt.figure()
    plt.plot(x, y, label="Exact solution")
    plt.plot(x_test, y_pred, color="black", label="NN prediction")
    plt.legend()
    plt.savefig("results_nn.png")
    plt.close()

    # print mse test loss
    mse = criterion(y_pred, y_test)
    print(f"MSE Test Loss: {mse.item()}")
