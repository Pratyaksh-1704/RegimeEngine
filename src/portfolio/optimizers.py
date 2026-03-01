import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list

class PortfolioOptimizers:
    """
    Collection of portfolio allocation algorithms.
    """
    
    @staticmethod
    def markowitz_mvo(expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_aversion: float = 1.0) -> np.ndarray:
        """
        Mean-Variance Optimization (Markowitz).
        Maximizes: w.T * mu - (risk_aversion / 2) * w.T * Sigma * w
        Subject to: sum(w) = 1, w >= 0
        """
        n_assets = len(expected_returns)
        
        def objective(w):
            port_ret = np.dot(w, expected_returns)
            port_var = np.dot(w.T, np.dot(cov_matrix, w))
            return (risk_aversion / 2) * port_var - port_ret
            
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_guess = np.ones(n_assets) / n_assets
        
        result = minimize(objective, init_guess, bounds=bounds, constraints=constraints)
        return result.x

    @staticmethod
    def min_expected_shortfall(returns_history: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Minimizes Expected Shortfall (CVaR) at significance level alpha.
        Historical simulation approach.
        """
        n_assets = returns_history.shape[1]
        
        def objective(w):
            port_returns = np.dot(returns_history, w)
            # Find the Value at Risk (VaR) threshold
            var_threshold = np.percentile(port_returns, alpha * 100)
            # Shortfall is the expected loss beyond VaR
            shortfall = port_returns[port_returns <= var_threshold]
            if len(shortfall) == 0:
                return -var_threshold
            return -np.mean(shortfall) # Return positive loss to minimize
            
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_guess = np.ones(n_assets) / n_assets
        
        result = minimize(objective, init_guess, bounds=bounds, constraints=constraints)
        return result.x

    @staticmethod
    def hierarchical_risk_parity(cov_matrix: pd.DataFrame) -> pd.Series:
        """
        Simplified Hierarchical Risk Parity (HRP) based on Lopez de Prado.
        """
        # Calculate correlation matrix
        vols = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(vols, vols)
        
        # Distance matrix (clip to avoid floating point issues like -1.0000000000000002)
        dist_matrix = np.sqrt(np.clip(0.5 * (1 - corr_matrix), 0, 1))
        
        # Enforce exact symmetry
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
        # Enforce exact zero diagonal
        np.fill_diagonal(dist_matrix.values, 0.0)
        
        # Clustering
        # We need condensed distance matrix for linkage
        import scipy.spatial.distance as ssd
        condensed_dist = ssd.squareform(dist_matrix)
        link = linkage(condensed_dist, method='single')
        
        # Sort items
        sort_ix = leaves_list(link)
        
        # Recursive bisection (simplified version -> Inverse Variance Allocation on sorted clusters)
        # For a full HRP, we would do recursive bisection.
        # Here we do a simplified cluster risk parity.
        w = pd.Series(1.0, index=cov_matrix.columns)
        
        def _get_cluster_var(cov, items):
            cov_ = cov.loc[items, items]
            ivp = 1.0 / np.diag(cov_)
            ivp /= ivp.sum()
            return np.dot(ivp, np.dot(cov_, ivp))
            
        def _recurse(items):
            if len(items) == 1:
                return
            split = len(items) // 2
            left = items[:split]
            right = items[split:]
            
            var_left = _get_cluster_var(cov_matrix, left)
            var_right = _get_cluster_var(cov_matrix, right)
            
            # Allocation factor
            alpha = 1.0 - var_left / (var_left + var_right)
            
            w[left] *= alpha
            w[right] *= (1 - alpha)
            
            _recurse(left)
            _recurse(right)
            
        _recurse(cov_matrix.columns[sort_ix])
        
        # Normalize
        w /= w.sum()
        return w

if __name__ == "__main__":
    np.random.seed(42)
    ret = np.random.randn(100, 3) * 0.01 + 0.001
    cov = pd.DataFrame(np.cov(ret.T), columns=["A", "B", "C"], index=["A", "B", "C"])
    mu = np.mean(ret, axis=0)
    
    print("MVO:", PortfolioOptimizers.markowitz_mvo(mu, cov.values))
    print("Min ES:", PortfolioOptimizers.min_expected_shortfall(ret))
    print("HRP:\n", PortfolioOptimizers.hierarchical_risk_parity(cov))
