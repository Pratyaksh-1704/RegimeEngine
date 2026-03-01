import logging
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.data_loader import DataLoader as DataFetcher
from src.features.feature_engineering import FeatureEngineer
from src.models.encoder import TCNEncoder
from src.models.trainer import TCNTrainer
from src.data.dataset import WindowedDataset, InferenceDataset
from src.models.hmm import RegimeHMM
from src.portfolio.allocator import RegimeConditionalAllocator
from src.backtest.pwfcv import Backtester

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, tickers=None, start_date="2010-01-01", end_date="2024-01-01"):
        if tickers is None:
            self.tickers = ["SPY", "TLT", "GLD"]
        else:
            self.tickers = tickers
            
        self.start_date = start_date
        self.end_date = end_date
        self.window_size = 21
        self.tcn_epochs = 50
        self.n_components = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def run(self):
        # 1. Data Ingestion
        fetcher = DataFetcher(self.tickers, self.start_date, self.end_date)
        # Using mock data for immediate run if yfinance fails or isn't connected
        try:
            raw_data = fetcher.fetch_data()
        except:
            logger.warning("Failed standard fetch, generating mock data for pipeline test.")
            dates = pd.date_range(self.start_date, self.end_date, freq='B')
            data = []
            val = np.array([100.0] * len(self.tickers))
            for i in range(len(dates)):
                vol = 0.01 if (i // 252) % 2 == 0 else 0.04
                ret = np.random.randn(len(self.tickers)) * vol + 0.0002
                val = val * (1 + ret)
                data.append(val.copy())
            raw_data = pd.DataFrame(data, index=dates, columns=self.tickers)
        # 2. Feature Engineering
        engineer = FeatureEngineer(raw_data)
        features = engineer.generate_features()
        
        # 3. Representation Learning
        train_dataset = WindowedDataset(features, window_size=self.window_size)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        input_dim = features.shape[1]
        tcn = TCNEncoder(input_size=input_dim, num_channels=[16, 32, 64], latent_dim=8)
        trainer = TCNTrainer(tcn, device=self.device)
        
        loss_history = trainer.train(train_loader, epochs=self.tcn_epochs)
        
        # Extract Latent Embeddings
        inf_dataset = InferenceDataset(features, window_size=self.window_size)
        inf_loader = DataLoader(inf_dataset, batch_size=64, shuffle=False)
        latent_z = trainer.extract_features(inf_loader).numpy()
        
        # 4. Regime Detection
        # Match dates of latent_z with returns
        valid_dates = inf_dataset.dates
        # Use first asset return as a proxy for volatility sorting
        first_asset = self.tickers[0]
        proxy_returns = features.loc[valid_dates, f'{first_asset}_log_return']
        
        hmm = RegimeHMM(n_components=self.n_components)
        hmm.fit(latent_z, proxy_returns)
        
        regimes = hmm.predict_regimes(latent_z)
        regime_series = pd.Series(regimes, index=valid_dates)
        
        logger.info(f"Regime counts:\n{regime_series.value_counts()}")
        
        # 5. Portfolio Allocation
        allocator = RegimeConditionalAllocator()
        
        # Calculate daily weights based on the regime
        # To avoid lookahead bias, we will use trailing 63 days for Cov calculation
        # and shift the regime map by 1 day
        returns_df = features[[f"{t}_log_return" for t in self.tickers]].loc[valid_dates]
        returns_df.columns = self.tickers
        
        weights_list = []
        for i, date in enumerate(valid_dates):
            if i < 63:
                # Not enough history for covariance
                weights_list.append(np.ones(len(self.tickers)) / len(self.tickers))
                continue
                
            current_regime = regime_series.iloc[i-1] if i > 0 else 'Transitional'
            
            lookback_ret = returns_df.iloc[i-63:i]
            cov = lookback_ret.cov() * 252
            
            w = allocator.allocate(current_regime, lookback_ret, cov)
            weights_list.append([w[t] for t in self.tickers])
            
        weights_df = pd.DataFrame(weights_list, index=valid_dates, columns=self.tickers)
        
        # 6. Backtest
        backtester = Backtester()
        port_val, metrics = backtester.run_backtest(returns_df, weights_df)
        
        logger.info(f"Backtest Metrics: {metrics}")
        
        return {
            'prices': raw_data,
            'features': features,
            'regimes': regime_series,
            'weights': weights_df,
            'portfolio_value': port_val,
            'metrics': metrics,
            'loss_history': loss_history,
            'latent_z': latent_z,
            'hmm': hmm,
            'returns': returns_df
        }

if __name__ == "__main__":
    pipeline = Pipeline(tickers=["SPY", "TLT"])
    results = pipeline.run()
