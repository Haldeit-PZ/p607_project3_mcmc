import numpy as np
from utils import jackknife


def get_mag(cfgs: np.ndarray):
    """
    Compute the mean and error of the magnetization.

    Parameters:
    - cfgs: np.ndarray : Array of configurations for the scalar field.

    Returns:
    - Tuple containing the mean and jackknife error of the magnetization.
    """
    # Compute the mean magnetization across all dimensions except the first
    axis = tuple([i + 1 for i in range(len(cfgs.shape) - 1)])
    return jackknife(cfgs.mean(axis=axis))


def get_abs_mag(cfgs: np.ndarray):
    """
    Compute the mean and error of the absolute magnetization.

    Parameters:
    - cfgs: np.ndarray : Array of configurations for the scalar field.

    Returns:
    - Tuple containing the mean and jackknife error of the absolute magnetization.
    """
    # Compute the mean absolute magnetization across all dimensions except the first
    axis = tuple([i + 1 for i in range(len(cfgs.shape) - 1)])
    return jackknife(np.abs(cfgs.mean(axis=axis)))


def get_chi2(cfgs: np.ndarray):
    """
    Compute the mean and error of the susceptibility.

    Parameters:
    - cfgs: np.ndarray : Array of configurations for the scalar field.

    Returns:
    - Tuple containing the mean and jackknife error of the susceptibility.
    """
    # Volume of the lattice, used for normalization
    V = np.prod(cfgs.shape[1:])
    # Compute the mean magnetization across all dimensions except the first
    axis = tuple([i + 1 for i in range(len(cfgs.shape) - 1)])
    mags = cfgs.mean(axis=axis)
    # Compute susceptibility as the variance of magnetization scaled by volume
    return jackknife(V * (mags ** 2 - mags.mean() ** 2))


def get_propagator(cfgs: np.ndarray):
    """
    Compute the propagator with jackknife errors for symmetric lattices.

    Parameters:
    - cfgs: np.ndarray : Array of configurations for the scalar field.

    Returns:
    - np.ndarray : Array with the distance, mean propagator, and error for each distance.
    """
    # Mean squared of the field to subtract the disconnected part
    mag_sq = np.mean(cfgs) ** 2
    propagator = []
    # Axes for averaging over all except the configuration axis
    axis = tuple([i + 1 for i in range(len(cfgs.shape) - 1)])

    # Loop over distances to calculate the propagator
    for i in range(1, cfgs.shape[1], 1):
        props = []

        # Calculate the propagator in all spatial directions (mu)
        for mu in range(len(cfgs.shape) - 1):
            # Roll configurations by distance `i` in direction `mu+1`
            props.append(np.mean(cfgs * np.roll(cfgs, i, mu + 1), axis=axis))

        # Average propagators across directions and subtract disconnected part
        props = np.array(props).mean(axis=0)
        prop_mean, prop_err = jackknife(props - mag_sq)
        propagator.append([i, prop_mean, prop_err])

    return np.array(propagator)
