from string_1D import String_1D
import matplotlib.pyplot as plt
import numpy as np

N = 10                  # string nodes
m = 1.0                 # mass
ang_freq = 0.01
separation = 1
step_size = 0.1         # step size for Metropolis-Hastings proposals

num_iterations = 5000
burn_in_fraction = 0.2


string1 = String_1D(N, m, ang_freq, separation, step_size)

print(f"Initial Equilibrium Positions: {string1.eq_pos}")
print(f"Initial Current Positions: {string1.curr_pos}")

samples, action_list = string1.run_mcmc(num_iterations=num_iterations, burn_in_fraction=burn_in_fraction)

string1.plot_ring()

plt.figure(figsize=(8, 4))
plt.plot(action_list)
plt.title(f"Action overtime")
plt.xlabel("Iterations")
plt.ylabel("Action")
plt.show()

# # checking node 0

# node = 0  
# positions_node_0 = [sample[node] for sample in samples]

# # plot the position of node 0 over iterations (after burn-in)
# plt.figure(figsize=(8, 4))
# plt.plot(positions_node_0)
# plt.title(f"Position of Node {node} over MCMC Iterations (after burn-in)")
# plt.xlabel("MCMC Iteration")
# plt.ylabel("Position")
# plt.show()