import numpy as np
import math

class String:
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
        np.append(self.eq_positions, self.eq_positions[0])

        # leave initial position as a copy  
        self.curr_pos = self.eq_pos


    def total_action(self):
        """
        get action for the whole system
        """
        S = 0.0
        for i in range(self.N - 1):
            a = self.separation
            # neighbour stretch term
            S += a * 0.5 * ((self.curr_pos[i + 1] - self.curr_pos[i]) / a) ** 2

            # self oscillation term
            S += 0.5 * self.ang_freq ** 2 * self.curr_pos[i] ** 2
        return S
    
    def local_action(self, node):
        """
        get local action
            node -- node number, index + 1
        """
        a = self.separation
        S = a * 0.5 * ((self.curr_pos[node + 1] - self.curr_pos[node]) / a) ** 2 + 0.5 * self.ang_freq ** 2 * self.curr_pos[node] ** 2

    def metropolis_step(self):
        """
        Perform a single Metropolis-Hastings update on a random lattice string node

        returns:
            whether the change was accepted or rejected
        """
        # find a new random node
        node = np.random.randint(0, self.N)

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

        for i in range(num_iterations):
            if self.metropolis_step():
                acceptance_count += 1

            # Store samples after burn-in
            if i >= burn_in:
                samples.append(self.curr_pos.copy())

        acceptance_rate = acceptance_count / num_iterations
        print(f'Acceptance Rate: {acceptance_rate:.2f}')
        return samples

