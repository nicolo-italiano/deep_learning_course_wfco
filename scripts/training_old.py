import pandas as pd
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.nn import init

import sys
import logging
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s', # '%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# -------------------- Load Data --------------------
data = torch.load('../data/5wt_dataset_1000_slsqp.pt')
logging.info("Data loaded")
X = data[0]
Y = data[1]

logging.info("Loaded X shape: %s", X.shape)
logging.info("Loaded Y shape: %s", Y.shape)


cases_per_layout = 12 * 9 * 46

# -------------------- Split dataset --------------------
def split_dataset(X, Y, cases_per_layout, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, shuffle=False):
    """
    Split X, Y by layouts into train/val/test sets.
    """
    n_layouts = X.shape[0] // cases_per_layout
    logging.info("Total layouts: %s", n_layouts)
    assert X.shape[0] % cases_per_layout == 0, "Dataset size mismatch!"

    # Layout indices
    layout_indices = torch.arange(n_layouts)
    if shuffle:
        layout_indices = layout_indices[torch.randperm(n_layouts)]

    # Compute split sizes
    n_train = int(train_ratio * n_layouts)
    n_val = int(val_ratio * n_layouts)

    train_layouts = layout_indices[:n_train]
    val_layouts = layout_indices[n_train:n_train+n_val]
    test_layouts = layout_indices[n_train+n_val:]

    def get_split(layouts):
        idx = torch.cat([torch.arange(l*cases_per_layout, (l+1)*cases_per_layout) for l in layouts])
        return X[idx], Y[idx]

    X_train, Y_train = get_split(train_layouts)
    X_val, Y_val = get_split(val_layouts)
    X_test, Y_test = get_split(test_layouts)

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)



def normalize_train_based(X_train, Y_train, X_val, Y_val, X_test, Y_test, skip_indices=None):
    """
    Normalize X and Y to [-1, 1] using train-based min/max.
    skip_indices: list of feature indices to leave unchanged.
    """
    if skip_indices is None:
        skip_indices = []

    skip_indices = torch.tensor(skip_indices, dtype=torch.long)
    all_indices = torch.arange(X_train.size(1))
    normalize_indices = all_indices[~torch.isin(all_indices, skip_indices)]

    # Compute min/max from training set
    min_vals = X_train[:, normalize_indices].min(dim=0).values
    max_vals = X_train[:, normalize_indices].max(dim=0).values
    y_min = Y_train.min()
    y_max = Y_train.max()

    def apply_norm(X, Y):
        X_norm = X.clone()
        finite_mask = ~torch.isnan(X[:, normalize_indices])
        
        X_norm[:, normalize_indices][finite_mask] = (
                    2 * (X[:, normalize_indices][finite_mask] - min_vals.repeat(X.size(0), 1)[finite_mask]) /
                    (max_vals.repeat(X.size(0), 1)[finite_mask] - min_vals.repeat(X.size(0), 1)[finite_mask]) - 1
                )

        Y_norm = 2 * (Y - y_min) / (y_max - y_min) - 1
        return X_norm, Y_norm

    X_train_norm, Y_train_norm = apply_norm(X_train, Y_train)
    X_val_norm, Y_val_norm = apply_norm(X_val, Y_val)
    X_test_norm, Y_test_norm = apply_norm(X_test, Y_test)

    return (X_train_norm, Y_train_norm), (X_val_norm, Y_val_norm), (X_test_norm, Y_test_norm)


# --- Usage ---
(X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_dataset(X, Y, cases_per_layout)

(X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = normalize_train_based(
    X_train, Y_train, X_val, Y_val, X_test, Y_test, skip_indices=[10, 11, 12, 13, 14]
)
logging.info("Dataset splitted and normalized.")

logging.info(X_train[:11])
logging.info(Y_train[:11])

PAD_VALUE = 999  # outside normalized range [-1, 1]
X_train[torch.isnan(X_train)] = PAD_VALUE
X_val[torch.isnan(X_val)] = PAD_VALUE
X_test[torch.isnan(X_test)] = PAD_VALUE


# Create masks (1 for valid, 0 for padded)
mask_train = (X_train != PAD_VALUE).float()
mask_val = (X_val != PAD_VALUE).float()
mask_test = (X_test != PAD_VALUE).float()


# -------------------- Model --------------------
class YawRegressionNet(nn.Module):
    def __init__(self, n_inputs=17, hidden_layers=[512, 256, 128], n_outputs=1, negative_slope=0.01):
        super(YawRegressionNet, self).__init__()
        layers = []
        in_features = n_inputs
        for h in hidden_layers:
            linear = nn.Linear(in_features, h)
            # init.xavier_normal_(linear.weight)
            init.kaiming_normal_(linear.weight, a=negative_slope, nonlinearity='leaky_relu')
            init.constant_(linear.bias, 0.0)
            layers.append(linear)
            # layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            layers.append(nn.Dropout(0.2))
            in_features = h
        out_layer = nn.Linear(in_features, n_outputs)
        init.kaiming_normal_(out_layer.weight, a=negative_slope, nonlinearity='leaky_relu')
        # init.xavier_normal_(out_layer.weight)
        init.constant_(out_layer.bias, 0.0)
        layers.append(out_layer)
        self.model = nn.Sequential(*layers)

    
    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask  # zero out padded features
        return self.model(x)



# -------------------- Training Setup --------------------
batch_size = 512
n_epochs = 10
learning_rate = 1e-3
hidden_layers = [256, 256, 256, 256]   # [1024, 1024, 512, 512, 256, 128, 64]
torch.manual_seed(42)

logging.info("Batch size: %s", batch_size)
logging.info("Number of epochs: %s", n_epochs)
logging.info("Learning rate: %s", learning_rate)


filename = f"{batch_size}_{learning_rate}_{n_epochs}_" + "x".join(map(str, hidden_layers))

logging.info("Hidden layers: %s", hidden_layers)
logging.info("Filename identifier: %s", filename)

# Prepare datasets with masks and loaders
train_dataset = TensorDataset(X_train, Y_train, mask_train)
val_dataset = TensorDataset(X_val, Y_val, mask_val)
test_dataset = TensorDataset(X_test, Y_test, mask_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda')   # if torch.cuda.is_available() else 'cpu'
logging.info("Device: %s", device)  # device can be a torch.device; it will be str()-formatted

model = YawRegressionNet(n_inputs=X_train.shape[1], hidden_layers=hidden_layers, n_outputs=1).to(device)
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_datapoints = X_train.shape[0]
logging.info(f"Ratio data points / parameters: {n_datapoints / n_parameters:.2f}")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Denormalization function
y_min, y_max = -25.0, 25.0
def denormalize(y_norm):
    return (((y_norm + 1) / 2) * (y_max - y_min)) + y_min

# Incremental RMSE calculation
def compute_rmse(loader, model, device):
    rmse_sum = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb, maskb in loader:
            xb, yb, maskb = xb.to(device), yb.to(device), maskb.to(device)
            y_pred = model(xb, maskb).squeeze()
            y_pred_deg = denormalize(y_pred)
            y_true_deg = denormalize(yb)
            rmse_sum += ((y_pred_deg - y_true_deg) ** 2).sum().item()
            count += yb.size(0)
    return (rmse_sum / count) ** 0.5


# Lists to store RMSE
train_rmse_deg, val_rmse_deg = [], []


# Early stopping parameters
patience = 10  # number of epochs to wait for improvement
best_val_rmse = float('inf')
epochs_no_improve = 0
best_model_state = None

# -------------------- TRAINING LOOP --------------------
for epoch in trange(n_epochs, desc="Training epochs", colour="blue"):
    model.train()
    total_train_loss = 0.0
    for xb, yb, maskb in train_loader:
        xb, yb, maskb = xb.to(device), yb.to(device), maskb.to(device)
        optimizer.zero_grad()
        y_pred = model(xb, maskb)
        loss = criterion(y_pred.squeeze(), yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()


    # ---- VALIDATION ----
    model.eval()
    
    rmse_val = compute_rmse(val_loader, model, device)
    rmse_train = compute_rmse(train_loader, model, device)  # optional, can use subset for speed

    train_rmse_deg.append(rmse_train)
    val_rmse_deg.append(rmse_val)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        logging.info(f"Epoch {epoch+1}/{n_epochs}: Train RMSE(deg): {rmse_train:.4f}, Val RMSE(deg): {rmse_val:.4f}")

    
    # ---- EARLY STOPPING CHECK ----
    if rmse_val < best_val_rmse:
        best_val_rmse = rmse_val
        best_model_state = model.state_dict()  # save best model
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        logging.info(f"Early stopping at epoch {epoch+1}. Best Val RMSE: {best_val_rmse:.4f}")
        break
    
# Load best model before evaluation
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# make filename directory if not exists
import os
os.makedirs(f"../results/figures/{filename}", exist_ok=True)


# -------------------- Plot --------------------
plt.figure()
plt.title("Training and Validation RMSE Loss")
plt.plot(train_rmse_deg, 'r', label='Train Loss')
plt.plot(val_rmse_deg, 'b', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('RMSE Loss (°)')
plt.legend()
plt.grid()
plt.tight_layout()
# Save the plot
plt.savefig(f"../results/figures/{filename}/{filename}.png", dpi=300)
logging.info(f"Plot saved as {filename}.png")



# -------------------- Scatter Plot of Predictions vs True Values --------------------
model.eval()
all_preds, all_true = [], []

with torch.no_grad():
    for xb, yb, maskb in train_loader:
        xb, yb, maskb = xb.to(device), yb.to(device), maskb.to(device)
        y_pred = model(xb, maskb).squeeze()
        all_preds.append(y_pred.cpu())
        all_true.append(yb.cpu())

# Concatenate all batches
all_preds = torch.cat(all_preds).numpy()
all_true = torch.cat(all_true).numpy()

# Compute R^2 score
from sklearn.metrics import r2_score
r2 = r2_score(all_true, all_preds)

# Plot
plt.figure()
plt.scatter(all_preds, all_true, s=2, alpha=0.5)
plt.xlabel("Normalized yaw prediction")
plt.ylabel("Normalized training data")
plt.title(f"Train Data $R^2$: {r2:.2f}")
plt.grid(True)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
# add y=x line black and dashed
plt.plot([-1, 1], [-1, 1], 'k--', lw=1)
plt.tight_layout()
plt.savefig(f"../results/figures/{filename}/{filename}_r2plot.png", dpi=300)
logging.info(f"Scatter plot saved as {filename}_scatter.png")


# -------------------- Scatter Plot on Test Set --------------------

model.eval()
all_preds_test, all_true_test = [], []

with torch.no_grad():
    for xb, yb, maskb in test_loader:  # <-- use test_loader here
        xb, yb, maskb = xb.to(device), yb.to(device), maskb.to(device)
        y_pred = model(xb, maskb).squeeze()
        all_preds_test.append(y_pred.cpu())
        all_true_test.append(yb.cpu())

# Concatenate all batches
all_preds_test = torch.cat(all_preds_test).numpy()
all_true_test = torch.cat(all_true_test).numpy()

# Compute R^2 score
from sklearn.metrics import r2_score
r2_test = r2_score(all_true_test, all_preds_test)

# Plot
plt.figure()
plt.scatter(all_preds_test, all_true_test, s=2, alpha=0.5)
plt.xlabel("Normalized yaw prediction")
plt.ylabel("Normalized test data")
plt.title(f"Test Data $R^2$: {r2_test:.2f}")
plt.grid(True)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.plot([-1, 1], [-1, 1], 'k--', lw=1)
plt.tight_layout()
plt.savefig(f"../results/figures/{filename}/{filename}_r2plot_test.png", dpi=300)
logging.info(f"Test scatter plot saved as {filename}_r2plot_test.png")

# -- -------------------- Calculate metrics on test data --------------------

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, max_error

# Denormalize predictions and true values
y_pred_deg = denormalize(all_preds_test)
y_true_deg = denormalize(all_true_test)

# Compute metrics
rmse_test = root_mean_squared_error(y_true_deg, y_pred_deg)  # RMSE
mae_test = mean_absolute_error(y_true_deg, y_pred_deg)
r2_test_denorm = r2_score(y_true_deg, y_pred_deg)
mape_test = (abs((y_true_deg - y_pred_deg) / y_true_deg).mean()) * 100  # in %
max_err_test = max_error(y_true_deg, y_pred_deg)

# Log results
logging.info(f"Test Metrics (De-normalized):")
logging.info(f"RMSE: {rmse_test:.4f}°")
logging.info(f"MAE: {mae_test:.4f}°")
logging.info(f"R²: {r2_test_denorm:.4f}")
logging.info(f"MAPE: {mape_test:.2f}%")
logging.info(f"Max Error: {max_err_test:.4f}°")



# Save the trained model
# torch.save(model.state_dict(), f"../models/yaw_regression_model_{filename}.pth")
# print(f"Model saved as yaw_regression_model_{filename}.pth")