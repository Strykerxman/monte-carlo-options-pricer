import numpy as np
import matplotlib.pyplot as plt

class StockPriceSimulator():

    def __init__(self, s0: float, mu: float, sigma: float, T: float, n_sims: int, n_steps: int):
        self.s0 = s0 # initial price
        self.mu = mu # mean
        self.sigma = sigma # volatility
        self.T = T # horizon time in years
        self.n_steps = n_steps 
        
        self.n_sims = n_sims
        
        self.rng = np.random.default_rng(42)
        #self.sample = np.random.normal(mu, sigma)

    def simulate_GBM(self, option_type: str = 'european', seed: int = 42) -> np.ndarray:
        np.random.seed(seed)

        if option_type not in ['european', 'asian', 'american']:
            raise ValueError("option_type must be either 'european' or 'asian'")
        
        if option_type == 'european':
            increments = np.random.normal(loc=(self.mu - 0.5 * self.sigma**2) * self.T, scale=self.sigma * np.sqrt(self.T), size=self.n_sims)
            S_paths = self.s0 * np.exp(increments)


            return S_paths
        
        elif option_type in ['asian', 'american']:

            dt = int(self.T / self.n_steps) # time increment

            paths = np.zeros((self.n_sims, self.n_steps + 1))
            paths[:, 0] = self.s0

            for t in range(1, self.n_steps + 1):
                Z = self.rng.standard_normal(self.n_sims)
                paths[:, t] = paths[:, t-1] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z)

            return paths
    



        