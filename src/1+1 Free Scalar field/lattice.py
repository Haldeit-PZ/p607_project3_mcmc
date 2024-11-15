import numpy as np
import copy
import emcee


class Lattice:
    def __init__(self, N, d, k, l):
        """
        Initialize the Lattice class with the following parameters:

        Parameters:
        - N (int) : Number of lattice points in each dimension.
        - d (int) : Number of dimensions of the lattice.
        - k (float) : Coupling constant for nearest-neighbor interaction.
        - l (float) : Coupling constant for the quartic term in the action.
        """
        self.N = N  # Lattice size in each dimension
        self.d = d  # Dimensionality of the lattice
        self.shape = [N for _ in range(d)]  # Shape of the lattice array
        self.k = k  # Nearest-neighbor coupling constant
        self.l = l  # Quartic coupling constant

        # Initialize the scalar field 'phi' with random values drawn from a normal distribution
        self.phi = np.random.randn(*self.shape)

        # Compute the initial action of the lattice
        self.action = self.get_action()

    def get_action(self):
        """
        Compute the total action for the lattice configuration.

        Returns:
        - action (float) : Sum of the action for the entire lattice.
        """
        # Start with local action (phi^2 and phi^4 terms)
        action = (1 - 2 * self.l) * self.phi ** 2 + self.l * self.phi ** 4

        # Add nearest-neighbor interaction terms
        for mu in range(self.d):
            action += -2. * self.k * self.phi * np.roll(self.phi, 1, mu)

        return action.sum()  # Return total action as a scalar

    def get_local_action(self, xyz):
        """
        Compute the local action at a specific lattice site.

        Parameters:
        - xyz (tuple) : Coordinates of the lattice site.

        Returns:
        - action (float) : Local action at the specified site.
        """
        # Compute local terms (phi^2 and phi^4) at the given site
        action = (1 - 2 * self.l) * self.phi[xyz] ** 2 + self.l * self.phi[xyz] ** 4

        # Add interactions with neighboring sites in each direction
        for mu in range(self.d):
            hop = np.zeros((self.d, 1), dtype=int)
            hop[mu, 0] = 1
            # Calculate coordinates of neighboring sites with periodic boundary conditions
            xyz_plus = tuple(map(tuple, ((np.array(xyz) + hop) % self.N)))
            xyz_minus = tuple(map(tuple, ((np.array(xyz) - hop) % self.N)))
            action += -2. * self.k * self.phi[xyz] * (self.phi[xyz_plus] + self.phi[xyz_minus])

        return action

    def get_drift(self):
        """
        Compute the drift term for each lattice site for stochastic processes.

        Returns:
        - drift (np.ndarray) : Array of drift values for each lattice site.
        """
        # Compute drift term (based on derivatives of the local action)
        drift = 2 * self.phi * (2 * self.l * (1 - self.phi ** 2) - 1)

        # Add contributions from neighboring sites in each direction
        for mu in range(self.d):
            drift += 2. * self.k * (np.roll(self.phi, 1, mu) + np.roll(self.phi, -1, mu))

        return drift

    def get_hamiltonian(self, chi, action):
        """
        Compute the Hamiltonian for the lattice configuration.

        Parameters:
        - chi (np.ndarray) : Conjugate momentum array.
        - action (float) : The action value for the current configuration.

        Returns:
        - hamiltonian (float) : The total Hamiltonian.
        """
        # Hamiltonian = kinetic energy (chi^2 term) + potential energy (action)
        return 0.5 * np.sum(chi ** 2) + action

    def metropolis(self, sigma=1.):
        """
        Perform a Metropolis update on a randomly selected lattice site.

        Parameters:
        - sigma (float) : Standard deviation of the random update for phi.

        Returns:
        - accepted (bool) : Whether the update was accepted or not.
        """
        # Select a random site for the update
        xyz = tuple(map(tuple, np.random.randint(0, self.N, (self.d, 1))))
        # Store initial phi value and local action at the selected site
        phi_0 = self.phi[xyz]
        S_0 = self.get_local_action(xyz)

        # Propose a new value for phi at the selected site
        self.phi[xyz] += sigma * np.random.randn()

        # Calculate the change in action (dS) from the update
        dS = self.get_local_action(xyz) - S_0

        # Accept or reject the new configuration based on Metropolis criterion
        if dS > 0:  # If action increases, accept with probability exp(-dS)
            if np.random.rand() >= np.exp(-dS):
                # Revert to the old phi value if the update is rejected
                self.phi[xyz] = phi_0
                return False  # Update rejected
        return True  # Update accepted

    def run_emcee(self, n_walkers=10, n_steps=1000):
        # Define the log-probability function for emcee
        def log_prob(phi_flat):
            self.phi = phi_flat.reshape(self.shape)  # Reshape into lattice shape
            return -self.get_action()  # Negative because emcee maximizes log-probability

        # Use diagonal initialization with small perturbations
        initial_pos = []
        for i in range(n_walkers):
            # Start each walker with all zeros and set one element to a distinct value
            base_pos = np.zeros(self.phi.flatten().shape)
            base_pos[i % base_pos.size] = 1.0  # Set a unique value for independence
            initial_pos.append(base_pos + 0.1 * np.random.randn(*base_pos.shape))  # Add small perturbation

        # Convert initial positions to a NumPy array
        initial_pos = np.array(initial_pos)

        # Initialize the sampler
        sampler = emcee.EnsembleSampler(n_walkers, self.phi.size, log_prob)

        # Run the MCMC chain
        sampler.run_mcmc(initial_pos, n_steps, progress=True)

        # Return the samples reshaped as lattice configurations
        samples = sampler.get_chain(discard=int(n_steps * 0.1), thin=10, flat=True)
        return samples.reshape((-1, *self.shape))  # Reshape each sample to lattice shape
