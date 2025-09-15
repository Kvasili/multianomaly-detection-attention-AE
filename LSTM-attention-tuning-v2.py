

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterSampler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random

# =============================
# Example Autoencoder (replace with yours)
# =============================


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# =============================
# Training loop
# =============================


def train_model(model, train_loader, val_loader, config, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    model.to(device)

    train_losses, val_losses = [], []
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

    return train_losses, val_losses, val_losses[-1]


# =============================
# Random Grid Search
# =============================
# Define parameter grid
param_grid = {
    "hidden_dim": [32, 64, 128, 256],
    "latent_dim": [4, 8, 16, 32],
    "dropout": [0.0, 0.2, 0.5],
    "lr": [1e-4, 1e-3, 1e-2],
    "batch_size": [32, 64, 128],
    "epochs": [30]  # keep fixed for fair comparison
}

# Randomly sample from grid
n_iter = 10  # number of random configs you want
param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))

# Example dummy dataset (replace with your data)
X = np.random.rand(1000, 20).astype(np.float32)  # 1000 samples, 20 features
train_data = TensorDataset(torch.tensor(X[:800]))
val_data = TensorDataset(torch.tensor(X[800:]))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []

for i, config in enumerate(param_list):
    print(f"\n=== Training model {i+1}/{n_iter} with params: {config} ===")

    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(
        val_data, batch_size=config["batch_size"], shuffle=False)

    model = Autoencoder(input_dim=X.shape[1],
                        hidden_dim=config["hidden_dim"],
                        latent_dim=config["latent_dim"],
                        dropout=config["dropout"])

    train_losses, val_losses, final_val = train_model(
        model, train_loader, val_loader, config, device)

    results.append((config, final_val))

    # Plot losses for this run
    plt.figure(figsize=(6, 6))
    plt.plot(range(1, config["epochs"] + 1),
             train_losses, label="Training Loss", linewidth=2)
    plt.plot(range(1, config["epochs"] + 1), val_losses,
             label="Validation Loss", linewidth=2, linestyle="--")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title(f"Run {i+1} - Loss Curve", fontsize=16)
    plt.legend(fontsize=12)
    plt.show()

# =============================
# Show best config
# =============================
best_config, best_val = min(results, key=lambda x: x[1])
print("\nBest config:", best_config)
print("Validation Loss:", best_val)
