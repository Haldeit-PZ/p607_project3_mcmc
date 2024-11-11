import numpy as np


def binned_jackknife(cfgs, bin_size):
    """
    Perform a binned jackknife analysis on sorted configurations.

    Parameters:
    - cfgs: 3D array of lattice configurations (num_samples, N, N)
    - bin_size: Number of consecutive configurations to remove in each bin

    Returns:
    - jackknife_means: Array of mean field values for each bin removal
    - jackknife_errors: Array of jackknife errors for each bin size
    """
    cfgs = np.array(cfgs)
    num_samples = cfgs.shape[0]
    jackknife_means = []
    jackknife_errors = []
    max_bin_size = num_samples // bin_size + 1
    # Loop over bin sizes from 1 up to total number of samples
    for k in range(1, max_bin_size):
        bin_means = []

        # Perform jackknife by removing `k * bin_size` configurations at a time
        for i in range(0, num_samples, k * bin_size):
            # Exclude `k * bin_size` consecutive configurations starting from `i`
            mask = np.ones(num_samples, dtype=bool)
            mask[i:i + k * bin_size] = False
            reduced_cfgs = cfgs[mask]

            # Compute the mean field value with excluded configurations
            bin_means.append(np.mean(reduced_cfgs))

        # Compute jackknife mean and error for this bin size
        mean = np.mean(bin_means)
        error = np.sqrt((len(bin_means) - 1) * np.mean((bin_means - mean) ** 2))

        jackknife_means.append(mean)
        jackknife_errors.append(error)

    return np.array(jackknife_means), np.array(jackknife_errors)
