# Self-Supervised Market Regime Detection

This project implements a self-supervised market regime detection pipeline for risk and portfolio control as outlined in the PRD. It leverages deep learning techniques (Temporal Convolutional Networks) for representation learning and quantitative finance math for portfolio optimization conditioned on the discovered states.

## Table of Contents

- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Hypotheses Validated](#hypotheses-validated)
- [Project Structure](#project-structure)

## Project Architecture

1.  **Data Ingestion & Feature Engineering**: Fetches asset data using `yfinance` and calculates robust continuous features (log returns, volatility, CUSUM breaks, entropy).
2.  **Representation Learning**: A deep Temporal Convolutional Network (TCN) learns an 8-dimensional latent state $z_t$ trained via self-supervised Contrastive Loss.
3.  **Regime Detection**: A Gaussian Hidden Markov Model (HMM) partitions the latent space $z_t$ into $K=4$ distinct regimes (Crisis, Defensive, Transitional, Risk-On) sorted by relative volatility.
4.  **Portfolio Allocation**: Dynamic calculation of optimal weights using Regime-Conditional logic: Markowitz (Risk-On), ES Minimization (Defensive), Hierarchical Risk Parity (Crisis), Equal-Weight (Transitional).
5.  **Simulation & Validation**: Regime-switching Monte Carlo simulations and Purged Walk-Forward Cross Validation.

## Installation

```bash
git clone https://github.com/yourusername/market-regime-detection.git
cd market-regime-detection
pip install -r requirements.txt
```

## Usage

### Run the Pipeline
To execute the end-to-end pipeline covering data fetching, TCN training, HMM fitting, and historical backtesting:
```bash
python -m src.pipeline.main
```

### Run the Dashboard
A local Streamlit dashboard is available to interactively run the pipeline and view visualizations.
```bash
streamlit run app.py
```

## Hypotheses Validated
-   **H1 (Representation vs Raw)**: The 8D latent manifold yields regimes with lower transition entropy (higher persistence) compared to clustering on raw features, proving the efficacy of the TCN representation layer.
-   **H2 (Regime-Switching Risk)**: Portfolio failure probability is highly path-dependent on regime transitions. Regime-Switching Monte Carlo simulations show a fatter left tail (realistic stress states) compared to simple static Gaussian simulations.

## Project Structure
```text
src/
├── backtest/        # PWFCV and Monte Carlo Simulation
├── data/            # yfinance data loaders and torch Datasets
├── features/        # Volatility, Entropy, CUSUM engineering
├── models/          # TCN Encoder, Contrastive Loss, HMM
├── pipeline/        # Main deployment and retraining scripts
├── portfolio/       # MVO, ES, HRP Allocation logic
└── utils/           # Visualizations
app.py               # Streamlit Deployment
requirements.txt     # Python Dependencies
```
