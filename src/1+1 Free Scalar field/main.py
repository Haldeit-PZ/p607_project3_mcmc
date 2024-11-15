import numpy as np
import copy
from tqdm import tqdm
import pylab as plt
from lattice import Lattice
from observables import *
from scipy.optimize import curve_fit

# Lattice parameters
N = 16  # Number of lattice points in each dimension
d = 2  # Number of dimensions
default_k = 0.9  # Default coupling constant for nearest-neighbor interactions
default_lambda = 0.12  # Default coupling constant for the quartic term in the action

# Thermalization and recording parameters
thermalization_steps = int(1E3)  # Number of steps for thermalization (burn-in)
recording_steps = int(1E6)  # Number of steps for recording (post-thermalization)
save_interval = 100  # Save configuration every N steps during recording

# Choose sampling method
use_emcee = True  # Set to True to use emcee, False for Metropolis


# Function to run a single simulation
def run_simulation(N, d, k, lamb, thermalization_steps, recording_steps, save_interval, use_emcee=False):
    lattice = Lattice(N, d, k, lamb)

    # If using emcee
    if use_emcee:
        print(f"Running simulation with emcee (k={k}, lambda={lamb})...\n")
        n_walkers = 1000
        n_steps = 10000
        cfgs = lattice.run_emcee(n_walkers=n_walkers, n_steps=n_steps)

        # Calculate magnetization for each configuration in emcee samples
        magnetizations_recording = np.mean(cfgs.reshape(-1, n_walkers, N, N), axis=(1, 2, 3))
        magnetizations_thermalization = np.array([])  # Not applicable with emcee
        acceptance_rate = None  # emcee doesn't use acceptance rate in the same way

    # If using hand-coded Metropolis algorithm
    else:
        print(f"Running Metropolis simulation with k={k}, lambda={lamb}...\n")
        cfgs = []
        magnetizations_thermalization = []  # Magnetization during thermalization
        magnetizations_recording = []  # Magnetization after thermalization
        n_accepted = 0  # Counter for accepted moves

        for i in tqdm(range(thermalization_steps + recording_steps)):
            accepted = lattice.metropolis()
            n_accepted += accepted

            # Record magnetization during thermalization
            if i < thermalization_steps:
                magnetizations_thermalization.append(lattice.phi.mean())

            # Start recording after thermalization
            if i >= thermalization_steps:
                if (i - thermalization_steps) % save_interval == 0:
                    cfgs.append(copy.deepcopy(lattice.phi))
                    magnetizations_recording.append(lattice.phi.mean())

        cfgs = np.array(cfgs)
        magnetizations_thermalization = np.array(magnetizations_thermalization)
        magnetizations_recording = np.array(magnetizations_recording)
        acceptance_rate = n_accepted / (thermalization_steps + recording_steps)

    return cfgs, magnetizations_thermalization, magnetizations_recording, acceptance_rate


# Run the default simulation
cfgs, magnetizations_thermalization, magnetizations_recording, acceptance_rate = run_simulation(
    N, d, default_k, default_lambda, thermalization_steps, recording_steps, save_interval, use_emcee=use_emcee
)

if acceptance_rate is not None:
    print("\nAcceptance rate:", acceptance_rate)

# OBSERVABLES
print("\nCalculating Observables...\n")
mag_mean, mag_err = get_mag(cfgs)
print("M =", mag_mean, "+/-", mag_err)

mag_abs_mean, mag_abs_err = get_abs_mag(cfgs)
print("|M| =", mag_abs_mean, "+/-", mag_abs_err)

chi2_mean, chi2_err = get_chi2(cfgs)
print("Chi^2 =", chi2_mean, "+/-", chi2_err)

# Magnetization Plot (During Thermalization) - Only applicable for Metropolis method
if not use_emcee:
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(magnetizations_thermalization)), magnetizations_thermalization, label="Magnetization")
    plt.xlabel("Steps (During Thermalization)")
    plt.ylabel("Magnetization")
    plt.title("Magnetization During Thermalization")
    plt.legend()
    plt.grid(True)
    plt.show()

# Magnetization Plot (Post-Thermalization)
plt.figure(figsize=(8, 6))
plt.plot(range(len(magnetizations_recording)), magnetizations_recording, label="Magnetization")
plt.xlabel("Steps (Post-Thermalization)")
plt.ylabel("Magnetization")
plt.title("Magnetization After Thermalization")
plt.legend()
plt.grid(True)
plt.show()

# PROPAGATOR
propagator = get_propagator(cfgs)
distances = propagator[:, 0]
prop_means = propagator[:, 1]
prop_errors = propagator[:, 2]

plt.figure(figsize=(8, 6))
plt.errorbar(distances, prop_means, yerr=prop_errors, fmt='o', capsize=4,
             label=f"Propagator (k={default_k}, lambda={default_lambda})")
plt.xlabel("Distance")
plt.ylabel("Propagator")
plt.title("Propagator vs Distance")
plt.legend()
plt.grid(True)
plt.show()

# COMPARISON: Propagator for Different k and lambda Values
k_values = [0.5, 0.9, 1.2]
lambda_values = [0.05, 0.12, 0.5]

plt.figure(figsize=(10, 8))
for k in k_values:
    for lamb in lambda_values:
        cfgs, _, _, _ = run_simulation(N, d, k, lamb, thermalization_steps, recording_steps, save_interval,
                                       use_emcee=use_emcee)
        propagator = get_propagator(cfgs)
        distances = propagator[:, 0]
        prop_means = propagator[:, 1]
        plt.plot(distances, prop_means, label=f"k={k}, lambda={lamb}")

plt.xlabel("Distance")
plt.ylabel("Propagator")
plt.title("Propagator Comparison for Varying k and lambda")
plt.legend()
plt.grid(True)
plt.show()
