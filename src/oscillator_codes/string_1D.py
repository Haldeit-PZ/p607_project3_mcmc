import numpy as np
import math
import matplotlib.pyplot as plt


class String_1D:
    def __init__(self, N, m, ang_freq, separation=1, step_size=0.1):
        """
        Initializing the 1d quantum mechanical harmonic oscillator string
            N - number of nodes
            m - mass
            ang_freq - global oscillating frequency
            separation - nodal separation
            step_size -  step size for metropolis
        """
        self.N = N
        self.m = m
        self.ang_freq = ang_freq
        self.separation = separation
        self.step_size = step_size

        # set initial, evenly spread locations of nodes
        self.eq_pos = np.arange(0, separation * N, separation)

        # the "N+1" node has position of the first node, the same node
        np.append(self.eq_pos, self.eq_pos[0])

        # leave initial position as a copy
        self.curr_pos = self.eq_pos

    def total_action(self):
        """
        get action for the whole system
        """
        S = 0.0
        for i in range(1, self.N):
            a = self.separation
            # neighbour stretch term
            S += a * 0.5 * ((self.curr_pos[i] - self.curr_pos[i - 1]) / a) ** 2

            # self oscillation term
            S += 0.5 * self.ang_freq**2 * self.curr_pos[i] ** 2
        return S

    def local_action(self, node):
        """
        get local action
            node -- node number, index + 1
        """
        a = self.separation
        S = (
            a * 0.5 * ((self.curr_pos[node + 1] - self.curr_pos[node]) / a) ** 2
            + 0.5 * self.ang_freq**2 * self.curr_pos[node] ** 2
        )
        return S

    def metropolis_step(self, neighbor_influence=0.5):
        """
        Perform a single Metropolis-Hastings update on a random lattice string node

        params:
            neighbour_influence - desides how much a neighbour is affected each step

        returns:
            whether the change was accepted or rejected
        """
        # find a new random node
        node = np.random.randint(0, self.N - 1)

        # find current position, action
        curr_pos = self.curr_pos[node]
        curr_S = self.local_action(node)

        # update to a new random location, calculate new action
        self.curr_pos[node] = curr_pos + np.random.normal(0, self.step_size)
        new_S = self.local_action(node)

        # action difference
        delta_S = new_S - curr_S

        # accept or reject
        if delta_S <= 0 or np.random.rand() < np.exp(-delta_S):
            # proposal accepted, now modify neighboring nodes as well
            left_neighbor = (node - 1) % self.N
            right_neighbor = (node + 1) % self.N

            # apply a smaller, correlated shift to the neighboring nodes
            shift = (self.curr_pos[node] - curr_pos) * neighbor_influence
            self.curr_pos[left_neighbor] += shift
            self.curr_pos[right_neighbor] += shift
            return True
        else:
            self.curr_pos[node] = curr_pos
            return False

    def run_mcmc(self, num_iterations, burn_in_fraction=0.2):
        """
        run MCMC sampling for the field on the string.
        params:
            num_iterations -- Total number of MCMC steps
            burn_in_fraction -- Fraction of steps to discard as burn-in
        returns:
            samples -- List of field configurations after burn-in
        """
        burn_in = int(num_iterations * burn_in_fraction)
        samples = []
        acceptance_count = 0

        action_list = []

        for i in range(num_iterations):
            if self.metropolis_step():
                acceptance_count += 1
                action = self.total_action()
                action_list.append(action)

            # Store samples after burn-in
            if i >= burn_in:
                samples.append(self.curr_pos.copy())

        acceptance_rate = acceptance_count / num_iterations
        print(f"Acceptance Rate: {acceptance_rate:.2f}")
        return samples, action_list

    def plot_ring(self):
        """
        Plots the 1D string as a circular ring of oscillators.
        """
        radius = 1.0

        angles = np.linspace(0, 2 * np.pi, self.N, endpoint=False)

        x = radius * np.cos(angles)
        y = radius * np.sin(angles)

        # adjust y positions based on current displacements for a 3D effect
        y_displaced = y + self.curr_pos * 0.1  # Scaling for visibility

        # plotting
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(
            np.append(x, x[0]), np.append(y_displaced, y_displaced[0]), "b-", lw=2
        )  # Ring edges
        ax.plot(x, y_displaced, "ro")

        ax.set_aspect("equal")
        ax.set_xlim(-radius * 1.5, radius * 1.5)
        ax.set_ylim(-radius * 1.5, radius * 1.5)

        ax.set_title("1D Quantum String Oscillators on a Circular Ring")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        plt.show()
