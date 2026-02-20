import numpy as np
import matplotlib.pyplot as plt

from src.simulator import StockPriceSimulator

class MonteCarloPricer():

    def __init__(self, simulator: StockPriceSimulator, risk_free_rate: float = 0.03):
        self.simulator: StockPriceSimulator = simulator
        self.risk_free_rate = risk_free_rate

    def price_contract(
            self,
            contract: str, 
            K: float, 
            T: float = 1,
            option_type: str = 'european', 
            model: str = 'gbm'):
        """
        Prices a European or Asian call or put option using Monte Carlo simulation.
        
        :param contract: The type of option contract to price ('call' or 'put').
        :type contract: str
        :param K: The strike price of the option.
        :type K: float
        :param T: The time to maturity of the option in years (default is 1).
        :type T: float
        :param option_type: The type of option to price ('european' or 'asian', default is 'european').
        :type option_type: str
        :param model: The stock price model to use for simulation ('gbm', 'abm', or 'student_t', default is 'gbm').
        :type model: str
        :return: A dictionary containing the estimated option price, future price, standard error, model used, and number of simulations.
        :rtype: dict
        """

        
        self._validate_inputs(K, T, option_type, model, contract=contract)
        self._verify_T(T)

        _, discount_payoffs, payoffs = self.discounted_payoffs(K, T, option_type, model, contract=contract)

        price = np.mean(discount_payoffs)

        std_error = np.std(discount_payoffs) / np.sqrt(self.simulator.n_sims)

        return {
            "option_price": float(price),
            "expiry_payoff": float(payoffs.mean()), 
            "std_error": float(std_error),
            "model": model,
            "n_sims": self.simulator.n_sims
        }
    
    def plot_price_convergence(self, K: float, T: float, option_type: str = 'european', model: str = 'gbm', contract: str = 'call'):
        self._validate_inputs(K, T, option_type, model, contract)
        self._verify_T(T)

        _, disc_payoffs, payoffs = self.discounted_payoffs(K, T, option_type, model, contract)
        price = np.mean(disc_payoffs)
        cumulative_avg = np.cumsum(disc_payoffs) / np.arange(1, self.simulator.n_sims + 1) # why do we divide by np.arange(1, n_sims + 1)? because we want the average up to that point, not the total sum.

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cumulative_avg)
        
        ax.set_xlabel("Number of Simulations")
        ax.set_ylabel("Cumulative Average Price")
        ax.set_title(f"Convergence of present contract value for {option_type} {contract} Option with {model} Model")
        ax.text(0.95, 0.3, f"Contract Price: {price:.4f}", transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')
        ax.axhline(payoffs.mean(), color='red', linestyle='--', label=f'future {contract} payoff: {payoffs.mean():.2f}')
        ax.legend()
        ax.grid(True)
        plt.show()

    def sim_model_paths(self, model: str):
        if model == 'gbm':
            paths = self.simulator.simulate_gbm()
        elif model == 'abm':
            paths = self.simulator.simulate_normal()
        elif model == 'student_t':
            paths = self.simulator.simulate_student_t()

        else: raise ValueError(f"model must be 'gbm', 'abm', or 'student_t'. Got '{model}'.")

        return paths

    def discounted_payoffs(self, K: float, T: float, option_type: str, model: str, contract: str):
        """
        Simulates stock price paths, calculates payoffs based on the option type and contract, and discounts them to present value.

        :param K: The strike price of the option.
        :type K: float
        :param T: The time to maturity of the option in years.
        :type T: float
        :param option_type: The type of option to price ('european' or 'asian').
        :type option_type: str
        :param model: The stock price model to use for simulation ('gbm', 'abm', or 'student_t').
        :type model: str
        :param contract: The type of option contract to price ('call' or 'put').
        :type contract: str
        :return: A tuple containing the discount factor, discounted payoffs, and undiscounted payoffs.
        :rtype: tuple
        """
        paths = self.sim_model_paths(model)

        payoffs = self._payoff_from_type(paths, K, option_type, contract)

        discount_factor = np.exp(-self.risk_free_rate * T)
        discounted_payoffs = discount_factor * payoffs

        return discount_factor, discounted_payoffs, payoffs
    
    def _payoff_from_type(self, paths: np.ndarray, K: float, option_type: str, contract: str):
        if option_type == 'european':
            if contract == 'call':
                payoffs = np.maximum(paths[:, -1] - K, 0)
            elif contract == 'put':
                payoffs = np.maximum(K - paths[:, -1], 0)

        elif option_type == 'asian':
            avg_prices = np.mean(paths[:, 1:], axis=1)

            if contract == 'call':
                payoffs = np.maximum(avg_prices - K, 0)
            elif contract == 'put':
                payoffs = np.maximum(K - avg_prices, 0)

        else: raise ValueError(f"option_type must be 'european' or 'asian'. Got '{option_type}'.")

        return payoffs

    def _validate_inputs(self, K: float, T: float, option_type: str, model: str, contract: str):
        if K <= 0:
            raise ValueError("Strike price K must be positive.")
        if T <= 0:
            raise ValueError("Time to maturity T must be positive.")
        if option_type not in ['european', 'asian']:
            raise ValueError(f"option_type must be 'european' or 'asian'. Got '{option_type}'.")
        if model not in ['gbm', 'abm', 'student_t']:
            raise ValueError(f"model must be 'gbm', 'abm', or 'student_t'. Got '{model}'.")
        if contract not in ['call', 'put']:
            raise ValueError(f"contract must be 'call' or 'put'. Got '{contract}'.")
    
    def _verify_T(self, T: float):
        if abs(self.simulator.T - T) > 1e-6:
            raise ValueError(f"Simulator T={self.simulator.T} does not match option T={T}")