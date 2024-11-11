import numpy as np
from utils import binned_jackknife

def get_corr_func(cfgs: np.ndarray):
    """
    Calculate the connected two-point correlation function with errors
    for symmetric lattices using a jackknife error estimation method.

    Parameters:
    - cfgs: 3D array of lattice configurations (num_samples, N, N),
            where num_samples is the number of MCMC samples
            and N is the lattice size (assumed square lattice).

    Returns:
    - corr_func: 2D array where each row contains [distance, mean correlation, error].
    """
    # Compute the square of the mean field value across all configurations
    mag_sq = np.mean(cfgs) ** 2
    corr_func = []

    # Define the axis to average over for correlations (all axes except the first)
    axis = tuple([i + 1 for i in range(len(cfgs.shape) - 1)])

    # Loop over distances to compute correlation at each separation
    for dist in range(1, cfgs.shape[1]):
        # List to hold correlation values for each direction (x and t)
        corrs = []

        # Loop over spatial and temporal directions to compute correlation
        for mu in range(len(cfgs.shape) - 1):
            # Shift the lattice by `dist` in direction `mu+1`, then multiply with original
            corrs.append(np.mean(cfgs * np.roll(cfgs, dist, mu + 1), axis=axis))

        # Average the correlation over directions and calculate the connected correlation
        corrs = np.array(corrs).mean(axis=0)
        corr_mean, corr_err = binned_jackknife(corrs - mag_sq, 1)

        # Append the distance, mean correlation, and error to results
        corr_func.append([dist, corr_mean, corr_err])

    return np.array(corr_func)


def calculate_mean_field(cfgs, bin_size):
    """
    Calculate the average field value with jackknife binned error estimation.

    Parameters:
    - cfgs: 3D array of lattice configurations (num_samples, N, N)
    - bin_size: Number of configurations per bin for jackknife resampling

    Returns:
    - avg_field: Mean field value across all configurations
    - avg_error: Jackknife error estimate for the mean field value
    """
    # Compute the mean field value for each configuration
    avg_field_values = np.mean(cfgs, axis=(1, 2))

    # Perform jackknife error estimation on the mean field values
    avg_field, avg_error = jackknife_binned(avg_field_values, bin_size)

    return avg_field, avg_error