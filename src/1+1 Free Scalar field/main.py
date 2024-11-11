from lattice import Lattice
import matplotlib.pyplot as plt
import numpy as np
from observables import calculate_mean_field
from utils import binned_jackknife

# Parameters
N = 10         # Lattice size (NxN square lattice)
m = 1.0         # Mass parameter
lamb = 0.1      # Interaction strength for φ⁴ theory (set to 0 for free theory)
step_size = 0.1 # Step size for Metropolis-Hastings proposals
num_iterations = 5000 # Total MCMC steps
burn_in_fraction = 0.2 # Fraction of steps to discard as burn-in

# Initialize the lattice
lattice = Lattice(N, m, lamb, step_size)

# Function to plot a lattice configuration
# Function to plot the lattice as a grid with annotated values
def plot_lattice_grid(phi, title="Lattice Configuration"):
    fig, ax = plt.subplots()
    ax.matshow(phi, cmap="coolwarm", alpha=0.2)
    for i in range(N):
        for j in range(N):
            c = phi[i, j]
            ax.text(j, i, f"{c:.2f}", va="center", ha="center")
    plt.xlabel('X')
    plt.ylabel('Time')
    plt.title(title)
    plt.show()

# Plot the initial configuration as a grid with annotated values
plot_lattice_grid(lattice.phi, title="Initial Lattice Configuration")
# Run MCMC simulation
samples = lattice.run_mcmc(num_iterations, burn_in_fraction)

# Define maximum bin size as half the number of samples
max_bin_size = len(samples) // 2

# Initialize lists to store mean field values and errors for each bin size
jackknife_means = []
jackknife_errors = []

# Loop over bin sizes from 1 up to max_bin_size
for bin_size in range(1, max_bin_size + 1):
    mean, error = binned_jackknife(samples, bin_size)
    print(mean, error)
    jackknife_means.append(mean)
    jackknife_errors.append(error)

# Plot mean field value and its error vs bin size
plt.errorbar(range(1, max_bin_size + 1), jackknife_means, yerr=jackknife_errors, fmt='o-', capsize=5)
plt.xlabel('Bin Size (Number of Configurations Removed)')
plt.ylabel('Mean Field Value')
plt.title('Mean Field Value vs Bin Size with Jackknife Error')
plt.show()
