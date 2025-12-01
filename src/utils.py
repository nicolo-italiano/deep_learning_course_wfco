import numpy as np


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