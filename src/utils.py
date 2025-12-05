'''
source utils for data processing, normalization, and evaluation metrics
'''

import numpy as np
import os
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import sys
import logging
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s', # '%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def _process_layout(turbine_x, turbine_y, rotor_diameter, spread=0.1, normalize=False, nearest_idx=0):
    """
    Computes normalized distances (dx, dy) from each turbine to its nearest or nth nearest upstream turbine
    that lies within the Jensen wake region.

    Parameters:
        turbine_x (ndarray): Array of x-coordinates of turbines for each wind direction (shape: [n_wd, n_turbines]).
        turbine_y (ndarray): Array of y-coordinates of turbines for each wind direction (shape: [n_wd, n_turbines]).
        rotor_diameter (float): Rotor diameter of the turbines.
        spread (float): Wake spread factor for Jensen model (default is 0.1).
        nearest_idx (int) or (list): 0 = nearest, 1 = second-nearest, etc.

    Returns:
        dx (ndarray): x-distance to selected upstream turbine.
        dy (ndarray): y-distance to selected upstream turbine.
    """
    # Compute pairwise x and y distances between turbines
    x_dists = turbine_x[:, np.newaxis, :] - turbine_x[:, :, np.newaxis]
    y_dists = turbine_y[:, np.newaxis, :] - turbine_y[:, :, np.newaxis]

    # Only consider upstream turbines
    x_dists[x_dists <= 0] = np.inf

    # Determine if a turbine lies within the Jensen wake of another
    in_Jensen_wake = np.abs(y_dists) < (spread * x_dists + rotor_diameter)
    x_dists[~in_Jensen_wake] = np.inf  # Exclude turbines outside the wake

    if isinstance(nearest_idx, int):
        if nearest_idx == 0:
            # Fast path for nearest turbine (original function)
            dx = np.min(x_dists, axis=2)
            dy_indices = np.argmin(x_dists, axis=2)
            dy = np.take_along_axis(y_dists, dy_indices[:, :, np.newaxis], axis=2)[:, :, 0]
        else:
            # Only sort distances if a higher-order nearest turbine is requested
            sorted_indices = np.argsort(x_dists, axis=2)
            nearest_idx_clipped = np.clip(nearest_idx, 0, x_dists.shape[2]-1)
            dx_indices = sorted_indices[:, :, nearest_idx_clipped]
            dx = np.take_along_axis(x_dists, dx_indices[:, :, np.newaxis], axis=2)[:, :, 0]
            dy = np.take_along_axis(y_dists, dx_indices[:, :, np.newaxis], axis=2)[:, :, 0]
            dy_indices = dx_indices

        if normalize:
            return dx.T / rotor_diameter, dy.T / rotor_diameter
        else:
            return dx.T, dy.T, dy_indices.T
        
    elif isinstance(nearest_idx, (list, tuple, np.ndarray)):
        # Multiple indices case
        sorted_indices = np.argsort(x_dists, axis=2)
        nearest_idx_clipped = np.clip(nearest_idx, 0, x_dists.shape[2]-1)
        dx_list, dy_list = [], []

        for idx in nearest_idx_clipped:
            dx_indices = sorted_indices[:, :, idx]
            dx_i = np.take_along_axis(x_dists, dx_indices[:, :, np.newaxis], axis=2)[:, :, 0]
            dy_i = np.take_along_axis(y_dists, dx_indices[:, :, np.newaxis], axis=2)[:, :, 0]
            dx_list.append(dx_i.T)
            dy_list.append(dy_i.T)

        dx_array = np.stack(dx_list, axis=0)  # shape: [len(nearest_idx), n_turbines, n_wd]
        dy_array = np.stack(dy_list, axis=0)

        if normalize:
            return dx_array / rotor_diameter, dy_array / rotor_diameter
        else:
            return dx_array, dy_array
        

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

    # logging.info("Y_min:", y_min, "Y_max:", y_max)

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


def denormalize(y_norm, y_max=25.0, y_min=-25.0):
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