import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

class PurgedKFold:
    """
    Purged Walk-Forward Cross Validation.
    Splits the data into overlapping intervals but 'purges' the overlapping
    periods from the training set to prevent data leakage in overlapping rolling windows.
    """
    def __init__(self, n_splits=5, purge_pct=0.01, embargo_pct=0.01):
        """
        Args:
            n_splits: Number of Walk-Forward folds.
            purge_pct: Percentage of total data length to purge before the test set.
            embargo_pct: Percentage of total data length to embargo after the test set.
        """
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        
    def split(self, X: pd.DataFrame, y=None, groups=None):
        """
        Yields train and test indices for Walk-Forward CV with purging and embargo.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate purge and embargo sizes
        purge_size = int(n_samples * self.purge_pct)
        embargo_size = int(n_samples * self.embargo_pct)
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for train_index, test_index in tscv.split(X):
            # The standard TimeSeriesSplit doesn't purge or embargo
            # We must apply purge and embargo logic
            
            # The test set starts at test_index[0] and ends at test_index[-1]
            # We purge the end of the train set
            train_end = test_index[0] - purge_size
            if train_end <= 0:
                continue # Skip if train set is too small after purging
                
            purged_train_index = np.arange(0, train_end)
            
            # For a strict Walk-Forward, we only use past to predict future.
            # So embargo isn't strictly needed for the train set *before* the test set,
            # but if we were doing a combinatorial CV (where test is in the middle),
            # we'd need to embargo the train set *after* the test set.
            # Since TimeSeriesSplit only has train before test, we just purge.
            
            yield purged_train_index, test_index

class Backtester:
    """
    Coordinates the historical backtest of the regime model and allocation strategies.
    """
    def __init__(self, initial_capital=10000.0):
        self.initial_capital = initial_capital
        
    def run_backtest(self, returns_df: pd.DataFrame, weights_df: pd.DataFrame):
        """
        Simulates the portfolio performance over time given historical returns and weights.
        Args:
            returns_df (pd.DataFrame): Asset log returns.
            weights_df (pd.DataFrame): Portfolio weights matching the returns index.
        """
        # Align indices
        common_idx = returns_df.index.intersection(weights_df.index)
        ret = returns_df.loc[common_idx]
        w = weights_df.loc[common_idx].shift(1).fillna(0) # Standard 1-period latency
        
        # Portfolio daily return
        port_ret = (np.exp(ret) - 1.0).multiply(w, axis=0).sum(axis=1) # Convert log return to simple
        
        # Cumulative return
        cum_ret = (1 + port_ret).cumprod()
        portfolio_value = self.initial_capital * cum_ret
        
        # Calculate metrics
        total_return = portfolio_value.iloc[-1] / self.initial_capital - 1
        annualized_return = (1 + total_return) ** (252 / len(port_ret)) - 1
        annualized_vol = port_ret.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': self._calculate_max_drawdown(portfolio_value)
        }
        
        return portfolio_value, metrics
        
    def _calculate_max_drawdown(self, portfolio_value: pd.Series):
        peak = portfolio_value.cummax()
        drawdown = (portfolio_value - peak) / peak
        return drawdown.min()

if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(level=logging.INFO)
    
    # Mock data
    X = pd.DataFrame(np.random.randn(1000, 5))
    cv = PurgedKFold(n_splits=3, purge_pct=0.05)
    
    for train, test in cv.split(X):
        print(f"Train: [{train[0]} - {train[-1]}] | Test: [{test[0]} - {test[-1]}]")
