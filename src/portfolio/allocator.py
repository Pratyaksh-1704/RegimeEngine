import pandas as pd
import numpy as np
import logging
from src.portfolio.optimizers import PortfolioOptimizers

logger = logging.getLogger(__name__)

class RegimeConditionalAllocator:
    """
    Allocates portfolio weights conditionally based on the detected market regime.
    """
    def __init__(self, regime_map=None):
        if regime_map is None:
            # Default mapping of Regime Name -> Optimizer Function
            self.regime_map = {
                'Risk-On': self._allocate_mvo,
                'Defensive': self._allocate_min_es,
                'Transitional': self._allocate_equal_weight,
                'Crisis': self._allocate_hrp
            }
        else:
            self.regime_map = regime_map
            
    def _allocate_mvo(self, returns: pd.DataFrame, cov: pd.DataFrame) -> np.ndarray:
        n = len(returns.columns)
        if n == 1:
            return np.array([1.0])
        logger.info("Allocating using Markowitz MVO (Risk-On Regime)")
        mu = returns.mean().values * 252
        return PortfolioOptimizers.markowitz_mvo(mu, cov.values, risk_aversion=3.0)
        
    def _allocate_min_es(self, returns: pd.DataFrame, cov: pd.DataFrame) -> np.ndarray:
        n = len(returns.columns)
        if n == 1:
            return np.array([1.0])
        logger.info("Allocating using Minimum Expected Shortfall (Defensive Regime)")
        recent_ret = returns.tail(252).values
        if len(recent_ret) == 0:
            return self._allocate_equal_weight(returns, cov)
        return PortfolioOptimizers.min_expected_shortfall(recent_ret, alpha=0.05)
        
    def _allocate_hrp(self, returns: pd.DataFrame, cov: pd.DataFrame) -> np.ndarray:
        n = len(returns.columns)
        if n == 1:
            return np.array([1.0])
        logger.info("Allocating using Hierarchical Risk Parity (Crisis Regime)")
        weights = PortfolioOptimizers.hierarchical_risk_parity(cov)
        return weights.values
        
    def _allocate_equal_weight(self, returns: pd.DataFrame, cov: pd.DataFrame) -> np.ndarray:
        logger.info("Allocating using Equal Weight (Transitional Regime)")
        n = len(returns.columns)
        return np.ones(n) / n

    def allocate(self, regime: str, returns: pd.DataFrame, cov: pd.DataFrame) -> dict:
        """
        Calculates optimal weights given the current market regime.
        Args:
            regime (str): Name of the current regime.
            returns (pd.DataFrame): Historical returns up to t.
            cov (pd.DataFrame): Covariance matrix at t.
        Returns:
            dict: Asset weights
        """
        if regime not in self.regime_map:
            logger.warning(f"Unknown regime '{regime}'. Defaulting to Equal Weight.")
            weights = self._allocate_equal_weight(returns, cov)
        else:
            alloc_func = self.regime_map[regime]
            weights = alloc_func(returns, cov)
            
        return {asset: w for asset, w in zip(returns.columns, weights)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ret = pd.DataFrame(np.random.randn(252, 4) * 0.01 + 0.0005, columns=['SPY', 'TLT', 'GLD', 'USO'])
    cov = ret.cov() * 252
    
    allocator = RegimeConditionalAllocator()
    print("Risk-On:", allocator.allocate('Risk-On', ret, cov))
    print("Crisis:", allocator.allocate('Crisis', ret, cov))
