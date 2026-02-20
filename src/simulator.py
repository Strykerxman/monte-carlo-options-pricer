import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, t

class StockPriceSimulator():

    def __init__(self, s0: float, mu: float, sigma: float, T: float, n_sims: int, n_steps: int, df: int = 5):
        self.s0 = s0 # initial price
        self.df = df # degrees of freedom for Student's t distribution
        self.mu = mu # mean
        self.sigma = sigma # volatility
        self.T = T # horizon time in years
        self.n_steps = n_steps 
        
        self.n_sims = n_sims
        
        self.rng = np.random.default_rng(42)
        np.random.seed(42)
        #self.sample = np.random.normal(mu, sigma)

    def simulate_gbm(self) -> np.ndarray:
        """
        Simulates stock price paths using Geometric Brownian Motion (GBM).

        :return: A 2D array of simulated stock price paths where each row corresponds to a simulation and each column corresponds to a time step
        :rtype: ndarray
        """
        paths = np.zeros((self.n_sims, self.n_steps + 1))
        paths[:, 0] = self.s0
        dt = self.T / self.n_steps # time increment

        Z = self.rng.standard_normal((self.n_sims, self.n_steps))
        log_returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
        paths[:, 1:] = self.s0 * np.exp(np.cumsum(log_returns, axis=1))
        
        return paths
    
    def simulate_normal(self) -> np.ndarray:
        """
        Simulates stock price paths using a normal distribution for increments (Arithmetic Brownian Motion).
        
        :return: A 2D array of simulated stock price paths where each row corresponds to a simulation and each column corresponds to a time step
        :rtype: ndarray
        """
        paths = np.zeros((self.n_sims, self.n_steps + 1))
        paths[:, 0] = self.s0
        dt = self.T / self.n_steps # time increment

        const_mu = self.mu * self.s0
        const_sig = self.sigma * self.s0

        Z = self.rng.standard_normal((self.n_sims, self.n_steps))

        increments = const_mu * dt + const_sig * np.sqrt(dt) * Z # represents dS_t
    
        paths[:, 1:] = self.s0 + np.cumsum(increments, axis=1)

        return paths
    
    def simulate_student_t(self, df: int = 5, seed: int = 42) -> np.ndarray:
        """
        Simulates stock price paths using a Student's t distribution for increments.

        :param df: Degrees of freedom for the Student's t distribution (default is 5).
        """
        paths = np.zeros((self.n_sims, self.n_steps + 1))
        paths[:, 0] = self.s0

        dt = self.T / self.n_steps # time increment

        Z = self.rng.standard_t(df, size=(self.n_sims, self.n_steps))

        Z = Z * np.sqrt((df - 2) / df) # scale to have variance 1

        log_returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z

        paths[:, 1:] = self.s0 * np.exp(np.cumsum(log_returns, axis=1))

        return paths

    def plot_paths_and_terminal_hist(self, show_paths: int = 100, gbm_paths: np.ndarray = None, abm_paths: np.ndarray = None, student_t_paths: np.ndarray = None):
        """
        Plot simulated paths and terminal histograms for only the provided methods.

        :param show_paths: The number of paths to show in the plot (default is 100).
        :type show_paths: int
        :param gbm_paths: A 2D array of simulated GBM paths (optional).
        :type gbm_paths: np.ndarray, optional
        :param abm_paths: A 2D array of simulated ABM paths (optional).
        :type abm_paths: np.ndarray, optional
        :param student_t_paths: A 2D array of simulated Student's t paths (optional).
        :type student_t_paths: np.ndarray, optional
        """
        available = []
        if gbm_paths is not None:
            available.append(("gbm", gbm_paths, "GBM Simulated Stock Price Paths", "GBM: Terminal Prices", "blue"))
        if abm_paths is not None:
            available.append(("abm", abm_paths, "ABM Simulated Stock Price Paths", "ABM: Terminal Prices", "orange"))
        if student_t_paths is not None:
            available.append(("student_t", student_t_paths, "Student's t Simulated Stock Price Paths", "Student's t: Terminal Prices", "green"))
        else: 
            raise ValueError("At least one of gbm_paths, abm_paths, or student_t_paths must be provided to plot_paths_and_terminal_hist()")
        
        n_figs = len(available)
        if n_figs == 0:
            raise ValueError("No path arrays provided to plot_paths_and_terminal_hist()")

        fig, axes = plt.subplots(2, n_figs, figsize=(4 * n_figs, 8))
        if n_figs == 1:
            axes = axes.reshape(2, 1)

        
        means = [None] * 3
        means[0] = np.mean(gbm_paths[:, -1]) if gbm_paths is not None else None
        means[1] = np.mean(abm_paths[:, -1]) if abm_paths is not None else None
        means[2] = np.mean(student_t_paths[:, -1]) if student_t_paths is not None else None

        for col, (_key, paths, title_paths, title_hist, color) in enumerate(available):
            paths_to_plot = paths[:show_paths]

            axes[0, col].plot(paths_to_plot.T, color=color, alpha=0.3)
            axes[0, col].set_title(title_paths)
            axes[0, col].set_ylabel('Stock Price (S)')

            axes[1, col].hist(paths_to_plot[:, -1], bins=30, density=True, alpha=0.6, color=color, edgecolor='black')
            axes[1, col].set_title(title_hist)
            axes[1, col].set_xlabel('Price ($)')
            axes[1, col].set_ylabel('Frequency')
            axes[1, col].axvline(means[col], color='red', linestyle='--', label=f'Mean: {means[col]:.2f}')
            axes[1, col].legend()
            

        plt.tight_layout()
        plt.show()

    def plot_log_returns(self, gbm_paths: np.ndarray = None, abm_paths: np.ndarray = None, student_t_paths: np.ndarray = None, time: str = 'terminal'):
        available = []
        if gbm_paths is not None:
            available.append(("gbm", gbm_paths, f"GBM Log Returns ({time})", "blue"))
        if abm_paths is not None:
            available.append(("abm", abm_paths, f"ABM Log Returns ({time})", "orange"))
        if student_t_paths is not None:
            available.append(("student_t", student_t_paths, f"Student's t Log Returns ({time})", "green"))

        n_figs = len(available)
        if n_figs == 0:
            raise ValueError("No path arrays provided to plot_log_returns()")

        fig, axes = plt.subplots(1, n_figs, figsize=(5 * n_figs, 5))
        if n_figs == 1:
            axes = [axes]

        for ax, (_key, paths, title, color) in zip(axes, available):
            if time == 'terminal':
                log_returns: np.ndarray = np.log(paths[:, -1] / self.s0)
                ax.hist(log_returns, bins=30, density=True, alpha=0.6, color=color, edgecolor='black')

                x = np.linspace(log_returns.min(), log_returns.max(), 100)

                if _key == "gbm":
                    ax.plot(x, norm.pdf(x, loc=self.mu * self.T, scale=self.sigma * np.sqrt(self.T)), color='red', linestyle='--', label='Normal PDF')
                
                elif _key == "abm":
                    ax.plot(x, norm.pdf(x, loc=self.mu * self.T, scale=self.sigma * np.sqrt(self.T)), color='red', linestyle='--', label='Normal PDF')
                
                elif _key == "student_t":
                    scale = self.sigma * np.sqrt(self.T) * np.sqrt((self.df - 2) / self.df)
                    ax.plot(x, t.pdf(x, df=self.df, loc=self.mu * self.T, scale=scale), color='red', linestyle='--', label="Student's t PDF")
            
            elif time == 'all':
                log_returns = np.diff(np.log(paths), axis=1)
                log_returns_flat = log_returns.flatten()
                ax.hist(log_returns_flat, bins=30, density=True, alpha=0.6, color=color, edgecolor='black')
            
            kurtosis = pd.Series(log_returns).kurtosis()
            skew = pd.Series(log_returns).skew()
            ax.text(0.35, 0.95, f'Skewness: {skew:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')
            ax.text(0.35, 0.90, f'Kurtosis: {kurtosis:.2f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')

            ax.set_title(title)
            ax.set_xlabel('Log Return')
            ax.set_ylabel('Density')

        plt.tight_layout()
        plt.show()


    def plot_paths(self, paths: np.ndarray, show_paths: int = 100):
        paths_to_plot = paths[:show_paths]
        
        time_grid = np.arange(self.n_steps + 1)
        frames = []
        
        for t in range(self.n_steps + 1):
            time_show = time_grid[:t+1]

            scatter_paths = [
                go.Scatter(
                    x=time_show,
                    y=paths_to_plot[i, :t+1],
                    mode='lines',
                    line=dict(color='#00ffff', width=1),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='none'
                )
                for i in range(show_paths)
            ]

            scatter_main = go.Scatter(
                x=time_show,
                y=paths_to_plot[0, :t+1],
                mode='lines',
                line=dict(color='#ff00ff', width=2),
                name='Sample Path'
            )

            frames.append(go.Frame(data=[*scatter_paths, scatter_main], name=f'frame{t}'))

        init_t = 0
        time_show = time_grid[:init_t+1]

        scatter_paths_init = [
            go.Scatter(
                x=time_show,
                y=paths_to_plot[i, :init_t+1],
                mode='lines',
                line=dict(color='#00ffff', width=1),
                opacity=0.3,
                showlegend=False,
                hoverinfo='none'
            )
            for i in range(show_paths)
        ]

        scatter_main_init = go.Scatter(
            x=time_show,
            y=paths_to_plot[0, :init_t+1],
            mode='lines',
            line=dict(color='#ff00ff', width=2)
        )

        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=(f"Simulated Stock Price Paths (T={self.T} years, S0={self.s0})",)
            )
        
        for s in scatter_paths_init:
            fig.add_trace(s, row=1, col=1)
        fig.add_trace(scatter_main_init, row=1, col=1)

        fig.frames = frames
        fig.update_layout(
            height=800, width=1200,
            title_text="Simulated Stock Price Paths",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False,
            updatemenus=[{
                'type': 'buttons',
                'x': 0.5, 'y': -0.2,
                'xanchor': 'center', 'yanchor': 'top',
                'direction': 'left',
                'showactive': False,
                'buttons': [{
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 15, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                }]
            }]
        )

        fig.update_xaxes(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128, 128, 128, 0.3)')

        fig.update_xaxes(title_text='Time (t)', range=[0, self.n_steps])
        fig.update_yaxes(title_text='Stock Price (S)', range=[0, np.max(paths_to_plot) * 1.1])

        fig.show()
        
        