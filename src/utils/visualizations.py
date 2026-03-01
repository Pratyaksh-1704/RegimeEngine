import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Visualizer:
    """ Generates PRD-specified visualizations. """
    def __init__(self, output_dir="outputs/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use a stylistic default
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_regime_shaded_returns(self, dates: pd.DatetimeIndex, prices: np.ndarray, regimes: np.ndarray, title="Regime-Shaded S&P 500"):
        """ Shade the underlying price series with the discovered regimes. """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, prices, color='black', linewidth=1, label="Price")
        
        # Color map for regimes
        colors = {'Risk-On': 'green', 'Defensive': 'blue', 'Transitional': 'orange', 'Crisis': 'red'}
        
        # We need to find contiguous segments to shade efficiently
        df = pd.DataFrame({'date': dates, 'regime': regimes})
        # For simplicity in this demo, we'll scatter the regimes on top of the price,
        # or shade the background.
        
        for regime_name, color in colors.items():
            mask = df['regime'] == regime_name
            if mask.any():
                # Scatter plot for current regime
                ax.scatter(df.loc[mask, 'date'], prices[mask], color=color, label=regime_name, s=10, alpha=0.6)
                
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        plt.tight_layout()
        
        fig_path = self.output_dir / "regime_shaded_returns.png"
        fig.savefig(fig_path)
        plt.close(fig)
        logger.info(f"Saved: {fig_path}")

    def plot_transition_matrix(self, transition_matrix: np.ndarray, regime_names: list):
        """ Heatmap of the HMM transition matrix. """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                    xticklabels=regime_names, yticklabels=regime_names, ax=ax)
        ax.set_title("HMM Transition Matrix")
        ax.set_xlabel("To Regime")
        ax.set_ylabel("From Regime")
        plt.tight_layout()
        
        fig_path = self.output_dir / "transition_matrix.png"
        fig.savefig(fig_path)
        plt.close(fig)
        logger.info(f"Saved: {fig_path}")

    def plot_mc_fan_chart(self, paths: np.ndarray, title="Monte Carlo Fan Chart"):
        """
        Plots the distribution of Monte Carlo simulated paths over time.
        paths shape: (n_paths, n_steps)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = np.arange(paths.shape[1])
        
        # Calculate percentiles
        p5 = np.percentile(paths, 5, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0) # Median
        p75 = np.percentile(paths, 75, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        
        ax.plot(steps, p50, color='blue', label='Median', linewidth=2)
        ax.fill_between(steps, p25, p75, color='blue', alpha=0.3, label='50% CI')
        ax.fill_between(steps, p5, p95, color='blue', alpha=0.1, label='90% CI')
        
        ax.set_title(title)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Portfolio Value")
        ax.legend()
        plt.tight_layout()
        
        fig_path = self.output_dir / "mc_fan_chart.png"
        fig.savefig(fig_path)
        plt.close(fig)
        logger.info(f"Saved: {fig_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    viz = Visualizer()
    
    # Mock transition matrix
    p = np.array([[0.9, 0.05, 0.03, 0.02],
                  [0.1, 0.8, 0.05, 0.05],
                  [0.2, 0.2, 0.5, 0.1],
                  [0.01, 0.09, 0.2, 0.7]])
    names = ['Risk-On', 'Defensive', 'Transitional', 'Crisis']
    viz.plot_transition_matrix(p, names)
    
    # Mock Fan Chart
    paths = np.random.lognormal(0.0001, 0.01, size=(1000, 252)).cumprod(axis=1) * 100
    viz.plot_mc_fan_chart(paths)
