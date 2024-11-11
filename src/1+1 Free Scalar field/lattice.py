import numpy as np
import math as ma
class Lattice:
    def __init__(self, N, m, lamb=0.1, step_size=0.1):
        """
        Initialize the lattice for both free and interacting scalar field theories.

        Parameters:
        - N, Lattice dimensions (spatial and temporal)
        - m: Mass parameter
        - lamb: Interaction strength for φ⁴ theory (default 0 for free theory)
        - step_size: Step size for the Metropolis-Hastings proposal
        """
        self.N = N
        self.m = m
        self.lamb = lamb
        self.step_size = step_size
        self.phi = np.random.normal(0, 0.1, (N, N))  # Initialize field

    def action(self):
        """
        Compute the action for the entire lattice configuration.

        Returns:
        - S: Total action of the current field configuration
        """
        S = 0.0
        for x in range(self.N):
            for t in range(self.N):
                # Periodic boundary conditions
                phi_xp = self.phi[(x + 1) % self.N, t]
                phi_tp = self.phi[x, (t + 1) % self.N]

                # Kinetic and mass terms
                S += 0.5 * ((phi_xp - self.phi[x, t]) ** 2 + (phi_tp - self.phi[x, t]) ** 2)
                S += 0.5 * self.m ** 2 * self.phi[x, t] ** 2

                # Interaction term for φ⁴ theory if lamb_ > 0
                if self.lamb > 0:
                    S += (self.lamb /ma.factorial(4)) * (self.phi[x, t]) ** 4
        return S

    def local_action(self, x, t):
        """
        Compute the local action around a specific lattice site (x, t).

        Parameters:
        - x, t: Coordinates of the lattice site

        Returns:
        - S_loc: Local action at site (x, t)
        """
        phi_xp = self.phi[(x + 1) % self.N, t]
        phi_xm = self.phi[(x - 1) % self.N, t]
        phi_tp = self.phi[x, (t + 1) % self.N]
        phi_tm = self.phi[x, (t - 1) % self.N]

        # Kinetic and mass terms
        S_loc = 0.5 * ((phi_xp - self.phi[x, t]) ** 2 + (phi_xm - self.phi[x, t]) ** 2 +
                       (phi_tp - self.phi[x, t]) ** 2 + (phi_tm - self.phi[x, t]) ** 2)
        S_loc += 0.5 * self.m ** 2 * self.phi[x, t] ** 2

        # Interaction term for φ⁴ theory if lamb_ > 0
        if self.lamb > 0:
            S_loc += (self.lamb /ma.factorial(4)) * (self.phi[x, t]) ** 4
        return S_loc

    def metropolis_step(self):
        """
        Perform a single Metropolis-Hastings update on a random lattice site.

        Returns:
        - accepted: Whether the proposal was accepted
        """
        x, t = np.random.randint(0, self.N), np.random.randint(0, self.N)
        phi_old = self.phi[x, t]
        current_action = self.local_action(x, t)

        # Propose a new field value for phi at (x, t)
        phi_new = phi_old + np.random.normal(0, self.step_size)
        self.phi[x, t] = phi_new

        # Compute new local action
        delta_S = self.local_action(x, t) - current_action

        # Metropolis acceptance criterion
        if delta_S <= 0 or np.random.rand() < np.exp(-delta_S):
            return True
        else:
            self.phi[x, t] = phi_old  # Revert if not accepted
            return False

    def run_mcmc(self, num_iterations, burn_in_fraction=0.2):
        """
        Run MCMC sampling for the field on the lattice.

        Parameters:
        - num_iterations: Total number of MCMC steps
        - burn_in_fraction: Fraction of steps to discard as burn-in

        Returns:
        - samples: List of field configurations after burn-in
        """
        burn_in = int(num_iterations * burn_in_fraction)
        samples = []
        acceptance_count = 0

        for i in range(num_iterations):
            if self.metropolis_step():
                acceptance_count += 1

            # Store samples after burn-in
            if i >= burn_in:
                samples.append(self.phi.copy())

        acceptance_rate = acceptance_count / num_iterations
        print(f'Acceptance Rate: {acceptance_rate:.2f}')
        return samples



