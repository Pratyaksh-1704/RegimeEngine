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
