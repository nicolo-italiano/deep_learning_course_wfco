import pandas as pd
import numpy as np
import torch
from tqdm import trange
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.nn import init

# -------------------- Load Data --------------------
data = torch.load('../data/dataset_yaw_1000.pt')
X = data['X'].float()
Y = data['Y'].float()
print("Loaded X shape:", X.shape)
print("Loaded Y shape:", Y.shape)

# -------------------- Parameters --------------------
n_per_turbine = 4  # d_x, d_y, ws_eff, ti_eff
n_turbines = 46
cases_per_layout = 360 * 11
n_layouts = X.shape[1] // cases_per_layout
print("Number of layouts:", n_layouts)

# Feature indices
ws_eff_idx = torch.tensor([n_per_turbine * i + 2 for i in range(n_turbines)])
ti_eff_idx = torch.tensor([n_per_turbine * i + 3 for i in range(n_turbines)])
global_idx = torch.tensor(n_turbines * n_per_turbine)
ws_idx = torch.tensor([n_turbines * n_per_turbine + 0])

# -------------------- Split dataset --------------------
n_train_layouts = int(.6 * n_layouts)
n_val_layouts = int(.2 * n_layouts)
n_test_layouts = int(.2 * n_layouts)

train_start = 0
train_end = n_train_layouts * cases_per_layout
val_start = train_end
val_end = val_start + n_val_layouts * cases_per_layout
test_start = val_end
test_end = test_start + n_test_layouts * cases_per_layout

X_train = X[:, train_start:train_end]
Y_train = Y[:, train_start:train_end]
X_val = X[:, val_start:val_end]
Y_val = Y[:, val_start:val_end]
X_test = X[:, test_start:test_end]
Y_test = Y[:, test_start:test_end]

# -------------------- Standardization --------------------
# ws_eff
mean_ws_eff = X_train[ws_eff_idx, :].mean()
std_ws_eff = X_train[ws_eff_idx, :].std() or 1.0
X_train[ws_eff_idx, :] = (X_train[ws_eff_idx, :] - mean_ws_eff) / std_ws_eff
X_val[ws_eff_idx, :] = (X_val[ws_eff_idx, :] - mean_ws_eff) / std_ws_eff
X_test[ws_eff_idx, :] = (X_test[ws_eff_idx, :] - mean_ws_eff) / std_ws_eff

# ti_eff
mean_ti_eff = X_train[ti_eff_idx, :].mean()
std_ti_eff = X_train[ti_eff_idx, :].std() or 1.0
X_train[ti_eff_idx, :] = (X_train[ti_eff_idx, :] - mean_ti_eff) / std_ti_eff
X_val[ti_eff_idx, :] = (X_val[ti_eff_idx, :] - mean_ti_eff) / std_ti_eff
X_test[ti_eff_idx, :] = (X_test[ti_eff_idx, :] - mean_ti_eff) / std_ti_eff

# global features
mean_global = X_train[global_idx, :].mean()
std_global = X_train[global_idx, :].std()
std_global[std_global == 0] = 1.0
X_train[global_idx, :] = (X_train[global_idx, :] - mean_global) / std_global
X_val[global_idx, :] = (X_val[global_idx, :] - mean_global) / std_global
X_test[global_idx, :] = (X_test[global_idx, :] - mean_global) / std_global

# Y standardization
mean_Y = Y_train.mean(dim=1, keepdim=True)
std_Y = Y_train.std(dim=1, keepdim=True)
std_Y[std_Y == 0] = 1.0
Y_train = (Y_train - mean_Y) / std_Y
Y_val = (Y_val - mean_Y) / std_Y
Y_test = (Y_test - mean_Y) / std_Y

# Transpose for DataLoader
X_train = X_train.T
Y_train = Y_train.T
X_val = X_val.T
Y_val = Y_val.T

# -------------------- Model --------------------
class YawRegressionNet(nn.Module):
    def __init__(self, n_inputs=185, hidden_layers=[512, 256, 128], n_outputs=46, negative_slope=0.01):
        super(YawRegressionNet, self).__init__()
        layers = []
        in_features = n_inputs
        for h in hidden_layers:
            linear = nn.Linear(in_features, h)
            init.xavier_normal_(linear.weight)
            init.constant_(linear.bias, 0.0)
            layers.append(linear)
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            # layers.append(nn.Dropout(0.2))
            in_features = h
        out_layer = nn.Linear(in_features, n_outputs)
        init.xavier_normal_(out_layer.weight)
        init.constant_(out_layer.bias, 0.0)
        layers.append(out_layer)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# -------------------- Training Setup --------------------
batch_size = 2048
n_epochs = 100
learning_rate = 1e-4
hidden_layers = [512, 256, 128, 64]   # [1024, 1024, 512, 512, 256, 128, 64]
filename = f"{batch_size}_{learning_rate}_small_dropout"
print("Batch size:", batch_size)
print("Number of epochs:", n_epochs)
print("Learning rate:", learning_rate)
print("Hidden layers:", hidden_layers)

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

model = YawRegressionNet(n_inputs=185, hidden_layers=hidden_layers, n_outputs=46).to(device)
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_datapoints = X_train.shape[0]
print(f"Ratio data points / parameters: {n_datapoints / n_parameters:.2f}")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# -------------------- Training Loop --------------------
train_mse, val_mse = [], []
train_rmse_deg, val_rmse_deg = [], []

for epoch in trange(n_epochs, desc="Epochs", colour="green"):
    # ----- TRAIN -----
    model.train()
    total_train_loss = 0.0
    batch_rmse_list = []

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # Batch-level RMSE in standardized units (optional for analysis)
        batch_rmse = torch.sqrt(loss).item()
        batch_rmse_list.append(batch_rmse)

    # Epoch-level MSE in standardized units
    mse_train_epoch = total_train_loss / len(train_loader)

    # ----- VALIDATION -----
    model.eval()
    total_val_loss = 0.0
    preds_val, targets_val = [], []
    preds_train, targets_train = [], []

    with torch.no_grad():
        # Full train set for RMSE in degrees
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds_train.append(model(xb).cpu())
            targets_train.append(yb.cpu())
            

        # Validation set
        for xb_val, yb_val in val_loader:
            xb_val, yb_val = xb_val.to(device), yb_val.to(device)
            preds_val.append(model(xb_val).cpu())
            targets_val.append(yb_val.cpu())
            total_val_loss += criterion(model(xb_val), yb_val).item()

    # Concatenate batches
    Y_train_pred = torch.cat(preds_train, dim=0)
    Y_val_pred   = torch.cat(preds_val, dim=0)
    Y_train_true = torch.cat(targets_train, dim=0)
    Y_val_true   = torch.cat(targets_val, dim=0)

    # ----- Unscale (undo standardization) for RMSE -----
    Y_train_true_deg = Y_train_true * std_Y.T + mean_Y.T
    Y_val_true_deg   = Y_val_true * std_Y.T + mean_Y.T
    Y_train_pred_deg = Y_train_pred * std_Y.T + mean_Y.T
    Y_val_pred_deg   = Y_val_pred * std_Y.T + mean_Y.T

    # ----- Epoch-level RMSE in degrees -----
    rmse_train_deg_epoch = torch.sqrt(((Y_train_true_deg - Y_train_pred_deg) ** 2).mean()).item()
    rmse_val_deg_epoch   = torch.sqrt(((Y_val_true_deg - Y_val_pred_deg) ** 2).mean()).item()

    # ----- Store results -----
    train_mse.append(mse_train_epoch)            # MSE in standardized units
    val_mse.append(total_val_loss / len(val_loader))
    train_rmse_deg.append(rmse_train_deg_epoch)  # RMSE in degrees
    val_rmse_deg.append(rmse_val_deg_epoch)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{n_epochs}: Train MSE: {mse_train_epoch:.4f}, "
              f"Val MSE: {val_mse[-1]:.4f}, Train RMSE(deg): {rmse_train_deg_epoch:.4f}, "
              f"Val RMSE(deg): {rmse_val_deg_epoch:.4f}")

# ----- Plot epoch-level MSE -----
plt.figure()
plt.plot(train_mse, 'r', label='Train MSE')
plt.plot(val_mse, 'b', label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE (standardized units)')
plt.legend()
plt.grid()
plt.savefig(f"../results/figures/training_validation_mse_{filename}.png", dpi=300)

# ----- Plot batch-level RMSE for this epoch -----
# plt.figure(figsize=(8,4))
# plt.plot(batch_rmse_list, 'r.-', label='Batch RMSE')
# plt.xlabel('Batch')
# plt.ylabel('RMSE (standardized units)')
# plt.title(f'Batch-level RMSE - Epoch {epoch+1}')
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig(f"../results/figures/batch_rmse_epoch_{epoch+1}_{filename}.png", dpi=300)
# plt.close()

# -------------------- Plot --------------------
plt.figure()
plt.plot(train_rmse_deg, 'r', label='Train Loss')
plt.plot(val_rmse_deg, 'b', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('RMSE Loss (°)')
plt.legend()
plt.grid()
# Save the plot
plt.savefig(f"../results/figures/training_validation_loss_{filename}.png", dpi=300)
print(f"Plot saved as training_validation_loss_{filename}.png")


# Save the trained model
# torch.save(model.state_dict(), f"../models/yaw_regression_model_{filename}.pth")
# print(f"Model saved as yaw_regression_model_{filename}.pth")