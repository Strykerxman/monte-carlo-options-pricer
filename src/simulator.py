import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

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
        
        paths = np.zeros((self.n_sims, self.n_steps + 1))
        paths[:, 0] = self.s0
        dt = self.T / self.n_steps # time increment

        Z = self.rng.standard_normal((self.n_sims, self.n_steps))
        log_returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
        paths[:, 1:] = self.s0 * np.exp(np.cumsum(log_returns, axis=1))
        
        return paths
    

    def plot_paths(self, paths: np.ndarray, option_type: str = 'european', show_paths: int = 100):
        paths_to_plot = paths[:show_paths]
        
        max_pdf = max(norm.pdf(0,0,np.sqrt(t)) for t in range(1, self.n_steps + 1)) + 0.1
        time_grid = np.arange(self.n_steps + 1)

        frames = []
        
        if option_type == 'european':
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

            fig.update_xaxes(title_text='Time (t) (days)', range=[0, self.n_steps])
            fig.update_yaxes(title_text='Stock Price ($S)', range=[0, np.max(paths_to_plot) * 1.1])

            fig.show()
        


        