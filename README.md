# 🏛️ RegimeEngine — Self-Supervised Market Regime Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://regimeengine.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **🚀 Live Demo:** [https://regimeengine.streamlit.app](https://regimeengine.streamlit.app)

A self-supervised market regime detection framework for risk-aware portfolio control, combining **Temporal Convolutional Networks (TCN)** with **Gaussian Hidden Markov Models (HMM)** for unsupervised regime discovery.

---

## ✨ Key Features

- **Self-Supervised Learning** — Contrastive TCN encoder learns regime-discriminative latent embeddings without labels
- **4 Market Regimes** — Risk-On, Defensive, Transitional, Crisis — mapped by realized volatility
- **Dynamic Portfolio Allocation** — MVO (Risk-On), Min-ES (Defensive), HRP (Crisis), Equal Weight (Transitional)
- **Ablation Study** — Rigorous 4-method × 5-metric comparison proving TCN superiority
- **Interactive Dashboard** — 7-tab Streamlit app with real-time analytics

---

## 🏗️ Architecture

```
Raw Prices → Feature Engineering → TCN Encoder → Gaussian HMM → Portfolio Allocator → Backtest
   (yfinance)   (Returns, Vol,       (Contrastive     (4 states,       (MVO/MinES/       (Risk
                  CUSUM, Entropy,      Self-Supervised   10 restarts,     HRP/EqWt)         Analytics)
                  Correlation)         → ℝ⁸ latent)     diag covar)
```

## 📦 Installation

```bash
git clone https://github.com/Pratyaksh-1704/RegimeEngine.git
cd RegimeEngine
pip install -r requirements.txt
```

## 🚀 Usage

### Run the Dashboard
```bash
streamlit run app.py
```

### Run the Pipeline CLI
```bash
python -m src.pipeline.main
```

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 Overview | Portfolio value (risk-shaded), rolling Sharpe, key metrics |
| 🧮 Features | Engineered signals: volatility, CUSUM, entropy, correlations |
| 🧠 TCN Encoder | Training loss curve, PCA latent space visualization |
| 🔬 HMM Regimes | Regime-shaded price history, transition matrix |
| 💼 Portfolio | Dynamic allocations, drawdown analysis |
| 🤖 Advisor | AI risk advisory: Buy / Hold / Sell recommendation |
| 🔬 Ablation Study | 4-method comparison across 5 unsupervised metrics |

## 🔬 Ablation Study Results

| Method | Silhouette ↑ | CH Index ↑ | DB Index ↓ | Stability ↑ | Pred. Validity ↑ |
|--------|-------------|-----------|-----------|------------|-----------------|
| M0: Raw Returns | 0.081 | 42.3 | 2.89 | 3.2 d | 0.12 |
| M1: Features HMM | 0.134 | 118.7 | 2.14 | 6.8 d | 0.31 |
| M2: PCA + HMM | 0.152 | 134.2 | 1.97 | 8.1 d | 0.38 |
| **M3: TCN + HMM (Ours)** | **0.203** | **186.4** | **1.52** | **14.6 d** | **0.52** |

## 📁 Project Structure

```
├── app.py                  # Streamlit dashboard (7 tabs)
├── requirements.txt        # Python dependencies
├── .streamlit/config.toml  # Dark theme configuration
├── paper/                  # LaTeX research paper (IEEE format)
│   ├── main.tex
│   └── part2.tex
└── src/
    ├── backtest/           # PWFCV and Monte Carlo simulation
    ├── data/               # yfinance loaders, torch Datasets
    ├── evaluation/         # Ablation study module
    ├── features/           # Volatility, Entropy, CUSUM engineering
    ├── models/             # TCN Encoder, Contrastive Loss, HMM
    ├── pipeline/           # End-to-end pipeline
    ├── portfolio/          # MVO, ES, HRP allocation logic
    └── utils/              # Visualization helpers
```

## 📄 Research Paper

A full IEEE two-column conference paper is included in `paper/`. It covers:
- Abstract, Introduction, Related Works (17 references)
- Proposed Method with mathematical formulations
- Experimental Results on SPY/TLT/GLD (2015–2024)
- Ablation Study & Comparison with published methods

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch (TCN, Contrastive Loss)
- **Regime Detection:** hmmlearn (Gaussian HMM)
- **Features:** NumPy, Pandas, SciPy
- **Visualization:** Plotly, Streamlit
- **Data:** yfinance API
- **Portfolio:** scikit-learn, custom optimizers

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built by [Pratham Gupta](https://github.com/Pratyaksh-1704)**
