import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class RegimeSwitchingMonteCarlo:
    """
    Simulates asset paths using a Regime-Switching model.
    Transitions are governed by a Markov transition matrix.
    Returns in each regime are drawn from a multivariate normal distribution.
    """
    def __init__(self, transition_matrix: np.ndarray, regime_params: dict):
        """
        Args:
            transition_matrix: Shape (K, K) where K is number of regimes.
            regime_params: Dict of regime index -> {'mu': array, 'cov': matrix}
        """
        self.P = transition_matrix
        self.params = regime_params
        self.n_regimes = len(transition_matrix)
        self.n_assets = len(regime_params[0]['mu'])
        
    def simulate(self, current_regime: int, n_steps: int, n_paths: int = 1000) -> dict:
        """
        Simulates future return paths.
        
        Args:
            current_regime: The starting regime index.
            n_steps: Number of time steps to simulate (e.g., 252 for 1 year).
            n_paths: Number of Monte Carlo paths.
            
        Returns:
            dict containing:
                'returns': shape (n_paths, n_steps, n_assets)
                'regimes': shape (n_paths, n_steps)
        """
        logger.info(f"Starting Monte Carlo Simulation: {n_paths} paths, {n_steps} steps.")
        
        # Pre-allocate arrays
        sim_returns = np.zeros((n_paths, n_steps, self.n_assets))
        sim_regimes = np.zeros((n_paths, n_steps), dtype=int)
        
        # Initialize
        sim_regimes[:, 0] = current_regime
        
        # Simulate regime paths
        for t in range(1, n_steps):
            # For each path, determine next regime based on transition probabilities
            # To optimize, we can do it vectorized but loop over regimes
            for r in range(self.n_regimes):
                paths_in_r = (sim_regimes[:, t-1] == r)
                n_in_r = paths_in_r.sum()
                if n_in_r > 0:
                    probs = self.P[r]
                    next_r = np.random.choice(self.n_regimes, size=n_in_r, p=probs)
                    sim_regimes[paths_in_r, t] = next_r
                    
        # Simulate returns conditioned on regimes
        for r in range(self.n_regimes):
            mask = (sim_regimes == r)
            n_samples = mask.sum()
            if n_samples > 0:
                mu = self.params[r]['mu']
                cov = self.params[r]['cov']
                draws = np.random.multivariate_normal(mu, cov, size=n_samples)
                sim_returns[mask] = draws
                
        return {
            'returns': sim_returns,
            'regimes': sim_regimes
        }
        
    def simulate_portfolio_paths(self, sim_returns: np.ndarray, weights: np.ndarray, initial_value: float = 1.0) -> np.ndarray:
        """
        Calculates cumulative portfolio value paths for static weights.
        For dynamic weights, a more complex backtester is needed.
        """
        # sim_returns shape: (paths, steps, assets)
        # weights shape: (assets,)
        port_returns = np.dot(sim_returns, weights) # shape (paths, steps)
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + port_returns, axis=1)
        return initial_value * cum_returns


class GBMSimulator:
    """
    Geometric Brownian Motion simulator for advanced Monte Carlo analysis.
    P(t+1) = P(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    """

    @staticmethod
    def simulate_gbm(start_price: float, mu_daily: float, sigma_daily: float,
                     n_steps: int = 30, n_paths: int = 1000,
                     seed: int | None = None) -> np.ndarray:
        """
        Returns array of shape (n_paths, n_steps+1) including the start price.
        """
        if seed is not None:
            np.random.seed(seed)
        dt = 1.0
        Z = np.random.standard_normal((n_paths, n_steps))
        log_returns = (mu_daily - 0.5 * sigma_daily**2) * dt + sigma_daily * np.sqrt(dt) * Z
        log_paths = np.cumsum(log_returns, axis=1)
        log_paths = np.hstack([np.zeros((n_paths, 1)), log_paths])
        return start_price * np.exp(log_paths)

    @staticmethod
    def fan_chart_stats(paths: np.ndarray) -> dict:
        """Compute percentile paths for fan charts."""
        return {
            'p5':     np.percentile(paths, 5, axis=0),
            'p25':    np.percentile(paths, 25, axis=0),
            'median': np.percentile(paths, 50, axis=0),
            'mean':   np.mean(paths, axis=0),
            'p75':    np.percentile(paths, 75, axis=0),
            'p95':    np.percentile(paths, 95, axis=0),
        }

    @staticmethod
    def compute_probabilities(paths: np.ndarray, start_price: float) -> dict:
        """Compute key probabilities from final prices."""
        finals = paths[:, -1]
        return {
            'P(gain)':  float(np.mean(finals > start_price) * 100),
            'P(loss)':  float(np.mean(finals < start_price) * 100),
            'P(+2%)':   float(np.mean(finals > start_price * 1.02) * 100),
            'P(+5%)':   float(np.mean(finals > start_price * 1.05) * 100),
            'P(+10%)':  float(np.mean(finals > start_price * 1.10) * 100),
            'P(-2%)':   float(np.mean(finals < start_price * 0.98) * 100),
            'P(-5%)':   float(np.mean(finals < start_price * 0.95) * 100),
            'P(-10%)':  float(np.mean(finals < start_price * 0.90) * 100),
            'expected': float(np.mean(finals)),
            'median':   float(np.median(finals)),
        }

    @staticmethod
    def compute_risk_metrics(paths: np.ndarray, start_price: float,
                             alpha: float = 0.05) -> dict:
        """VaR and CVaR from simulated final prices."""
        finals = paths[:, -1]
        pnl = (finals - start_price) / start_price
        var_threshold = np.percentile(pnl, alpha * 100)
        cvar = np.mean(pnl[pnl <= var_threshold]) if np.any(pnl <= var_threshold) else var_threshold
        return {
            'VaR_95':  float(-var_threshold * 100),
            'CVaR_95': float(-cvar * 100),
            'VaR_price':  float(start_price * (1 + var_threshold)),
            'CVaR_price': float(start_price * (1 + cvar)),
        }

    @staticmethod
    def density_matrix(paths: np.ndarray, n_price_bins: int = 50) -> tuple:
        """
        Build a 2D density matrix for the heatmap.
        Returns (density, price_edges, days).
        """
        n_steps = paths.shape[1]
        p_min = np.percentile(paths, 1)
        p_max = np.percentile(paths, 99)
        price_edges = np.linspace(p_min, p_max, n_price_bins + 1)
        price_mids = 0.5 * (price_edges[:-1] + price_edges[1:])
        density = np.zeros((n_price_bins, n_steps))
        for t in range(n_steps):
            counts, _ = np.histogram(paths[:, t], bins=price_edges)
            total = counts.sum()
            density[:, t] = counts / total if total > 0 else counts
        days = np.arange(n_steps)
        return density, price_mids, days

    @staticmethod
    def sensitivity_grid(start_price: float,
                         mu_base: float, sigma_base: float,
                         n_steps: int = 30, n_paths: int = 2000) -> list:
        """
        Run a 3x3 grid of simulations varying sigma and mu.
        Returns list of 9 dicts with keys: label, mu, sigma, stats, probs.
        """
        sigma_scenarios = [
            ('Low σ',  sigma_base * 0.6),
            ('Base σ', sigma_base),
            ('High σ', sigma_base * 1.6),
        ]
        mu_scenarios = [
            ('Bear μ',    mu_base - abs(mu_base) * 2),
            ('Neutral μ', mu_base),
            ('Bull μ',    mu_base + abs(mu_base) * 2),
        ]
        results = []
        for s_name, sigma in sigma_scenarios:
            for m_name, mu in mu_scenarios:
                paths = GBMSimulator.simulate_gbm(
                    start_price, mu, sigma, n_steps, n_paths)
                stats = GBMSimulator.fan_chart_stats(paths)
                probs = GBMSimulator.compute_probabilities(paths, start_price)
                results.append({
                    'label': f"{s_name} · {m_name}",
                    'mu': mu, 'sigma': sigma,
                    'stats': stats, 'probs': probs, 'paths': paths,
                })
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = np.array([[0.9, 0.1], [0.2, 0.8]])
    params = {
        0: {'mu': np.array([0.001]), 'cov': np.array([[0.0001]])}, # Low risk
        1: {'mu': np.array([-0.002]), 'cov': np.array([[0.0009]])} # High risk
    }
    
    mc = RegimeSwitchingMonteCarlo(p, params)
    res = mc.simulate(current_regime=0, n_steps=252, n_paths=100)
    print("Returns shape:", res['returns'].shape)
    print("Regimes shape:", res['regimes'].shape)
    
    paths = mc.simulate_portfolio_paths(res['returns'], weights=np.array([1.0]))
    print("Paths shape:", paths.shape)
