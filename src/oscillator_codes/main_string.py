from string_1D import String_1D
import matplotlib.pyplot as plt
import numpy as np
import emcee
import string_1D

N = 10  # string nodes
m = 1.0  # mass
ang_freq = 0.01
separation = 1
step_size = 0.1  # step size for Metropolis-Hastings proposals

num_iterations = 5000
burn_in_fraction = 0.4


string1 = String_1D(N, m, ang_freq, separation, step_size)

print(f"Initial Equilibrium Positions: {string1.eq_pos}")
print(f"Initial Current Positions: {string1.curr_pos}")

samples, action_list = string1.run_mcmc(
    num_iterations=num_iterations, burn_in_fraction=burn_in_fraction
)

string1.plot_ring()

plt.figure(figsize=(8, 4))
plt.plot(action_list)
plt.title(f"Action overtime")
plt.xlabel("Iterations")
plt.ylabel("Action")
plt.show()


def log_probability(curr_pos, string):
    """
    Calculate the log-probability of the current string configuration.

    Parameters:
        curr_pos -- Array of node positions (sample)
        string -- Instance of the String_1D class

    Returns:
        log_prob -- Log-probability of the configuration
    """
    # Update the string instance with the current configuration
    string.curr_pos = curr_pos
    # Calculate action for the current configuration
    S = string.total_action()
    # Return log-probability, which is -S for Metropolis-Hastings
    return -S


n_walkers = 2 * N
n_params = N
# Initial positions for each walker, perturb around initial `curr_pos`
initial_positions = [
    string1.curr_pos + 0.01 * np.random.randn(N) for _ in range(n_walkers)
]
# Initialize the emcee sampler
sampler = emcee.EnsembleSampler(n_walkers, n_params, log_probability, args=[string1])
sampler_2 = emcee.EnsembleSampler(n_walkers, n_params, log_probability, args=[string1])
# Run the MCMC chain with `emcee`
emcee_samples = sampler.run_mcmc(initial_positions, num_iterations, progress=True)
emcee_samples_2 = sampler_2.run_mcmc(initial_positions, num_iterations, progress=True)

# Flatten the chain, discarding the burn-in samples
burn_in = int(num_iterations * burn_in_fraction)
emcee_flat_samples = sampler.get_chain(discard=burn_in, flat=True)
emcee_flat_samples_2 = sampler_2.get_chain(discard=burn_in, flat=True)


# Convert `String_1D` samples to a similar format
string_samples = np.array(samples)  # Convert list of samples to array

node_index = 1  # Choose a node to inspect

plt.hist(emcee_flat_samples[:, node_index], bins=30, alpha=0.5, label="emcee")
plt.hist(string_samples[:, node_index + 2], bins=30, alpha=0.5, label="String_1D")
plt.xlabel(f"Position of Node {node_index}")
plt.ylabel("Frequency")
plt.legend()
plt.title(f"Comparison of Position Distributions for Node {node_index}")
plt.show()
