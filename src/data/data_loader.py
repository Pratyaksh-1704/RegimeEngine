import yfinance as yf
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Downloads historical market data from Yahoo Finance.
    Robust to both the old yfinance API ('Adj Close') and
    the new yfinance >=0.2 API which uses a MultiIndex with 'Close'.
    """

    def __init__(self, tickers: list, start_date: str, end_date: str,
                 data_dir: str = 'data/raw'):
        self.tickers    = tickers
        self.start_date = start_date
        self.end_date   = end_date
        self.data_dir   = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def fetch_data(self) -> pd.DataFrame:
        logger.info(f"Fetching data for {self.tickers} "
                    f"from {self.start_date} to {self.end_date}")
        try:
            raw = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,   # gives adjusted "Close" directly, removes Adj Close
                progress=False,
            )

            # ── Column resolution ──────────────────────────────────────
            if isinstance(raw.columns, pd.MultiIndex):
                # New yfinance: (price_type, ticker)
                # After auto_adjust=True the best column is 'Close'
                preferred = ['Close', 'Adj Close']
                chosen = None
                for p in preferred:
                    if p in raw.columns.get_level_values(0):
                        chosen = p
                        break
                if chosen is None:
                    chosen = raw.columns.get_level_values(0)[0]
                    logger.warning(f"Using fallback price column: '{chosen}'")
                df = raw[chosen].copy()

                # If only one ticker, ensure we have that name as a column
                if df.ndim == 1:
                    df = df.to_frame(name=self.tickers[0])

            else:
                # Old yfinance or single-ticker without MultiIndex
                preferred = ['Adj Close', 'Close']
                chosen = None
                for p in preferred:
                    if p in raw.columns:
                        chosen = p
                        break
                if chosen is None:
                    chosen = raw.columns[0]
                    logger.warning(f"Using fallback price column: '{chosen}'")

                if len(self.tickers) == 1:
                    df = raw[[chosen]].copy()
                    df.columns = [self.tickers[0]]
                else:
                    df = raw[[chosen]].copy()

            # ── Clean ─────────────────────────────────────────────────
            df = df.ffill().dropna()

            if df.empty:
                raise ValueError(
                    f"No valid data returned for tickers {self.tickers} "
                    f"over {self.start_date} → {self.end_date}."
                )

            csv_path = self.data_dir / 'market_data.csv'
            df.to_csv(csv_path)
            logger.info(f"Fetched {len(df)} rows → {csv_path}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise


# ── CLI test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader(
        tickers=["SPY", "TLT", "GLD"],
        start_date="2015-01-01",
        end_date="2024-12-31"
    )
    df = loader.fetch_data()
    print(df.head())
    print("Shape:", df.shape)
