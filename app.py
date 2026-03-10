"""
RegimeEngine – Institutional-Grade Market Regime Detection Dashboard
Design System v1.0: Applied from design_system.md
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, time as dtime
from sklearn.decomposition import PCA
import logging

# ─────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="RegimeEngine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
# Design System CSS
# Design tokens match design_system.md exactly
# ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design Tokens ────────────────────────────────────────────── */
:root {
  /* Background layers */
  --bg-deep:       #060B14;
  --bg-base:       #0B1220;
  --bg-surface:    #111C2D;
  --bg-elevated:   #162236;
  --bg-overlay:    #1D2E45;

  /* Text */
  --text-primary:   #E2EAF4;
  --text-secondary: #7A9BB5;
  --text-muted:     #3D5A73;
  --text-mono:      #A8C7E8;

  /* Semantic / Risk */
  --risk-low:           #22C993;
  --risk-medium:        #E8A838;
  --risk-high:          #E84040;

  /* Accent */
  --accent:             #3B82F6;
  --accent-hover:       #1E40AF;
  --accent-glow:        rgba(59,130,246,0.12);

  /* Borders */
  --border-subtle:      rgba(59,130,246,0.10);
  --border-active:      rgba(59,130,246,0.30);
  --border-divider:     rgba(255,255,255,0.05);

  /* Chart palette */
  --c1: #3B82F6;
  --c2: #22C993;
  --c3: #A78BFA;
  --c4: #E8A838;
  --c5: #E84040;
  --c6: #2DD4BF;
  --c7: #FB7185;
  --c8: #86EFAC;
}

/* ── Global Reset ─────────────────────────────────────────────── */
*, html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  box-sizing: border-box;
}

/* ── Canvas ───────────────────────────────────────────────────── */
.stApp {
  background: var(--bg-deep);
  color: var(--text-primary);
}

/* ── Sidebar ──────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--bg-base);
  border-right: 1px solid var(--border-subtle);
}
section[data-testid="stSidebar"] * {
  color: var(--text-secondary) !important;
}
section[data-testid="stSidebar"] label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--text-muted) !important;
}

/* ── Metric Cards ─────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 18px 20px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.5);
  transition: border-color 150ms cubic-bezier(0.4,0,0.2,1),
              transform 150ms cubic-bezier(0.4,0,0.2,1);
}
[data-testid="metric-container"]:hover {
  border-color: var(--border-active);
  transform: translateY(-2px);
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 22px !important;
  font-weight: 600 !important;
  color: var(--text-mono) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  font-size: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 1.5px !important;
  color: var(--text-muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
}

/* ── Run Button ───────────────────────────────────────────────── */
div.stButton > button {
  background: linear-gradient(135deg, #1d4ed8 0%, #0d9488 100%);
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 13px 28px;
  font-weight: 700;
  font-size: 14px;
  letter-spacing: 0.3px;
  width: 100%;
  box-shadow: 0 4px 16px rgba(29,78,216,0.3);
  transition: transform 150ms cubic-bezier(0.4,0,0.2,1),
              box-shadow 150ms cubic-bezier(0.4,0,0.2,1);
}
div.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 28px rgba(29,78,216,0.45);
}

/* ── Tabs ─────────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  border-bottom: 2px solid transparent !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 12px 18px !important;
  transition: color 180ms ease, border-color 180ms ease !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}
button[data-baseweb="tab"]:hover {
  color: var(--text-secondary) !important;
  background: rgba(59,130,246,0.04) !important;
}

/* ── Section Titles ───────────────────────────────────────────── */
.st-section {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--text-muted);
  margin: 24px 0 10px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border-divider);
  display: flex;
  align-items: center;
  gap: 8px;
}

/* ── Panel Card ───────────────────────────────────────────────── */
.re-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 12px;
}

/* ── Advisor Boxes ────────────────────────────────────────────── */
.adv-buy   { background: linear-gradient(135deg,#012e1f,#024d33); border:1px solid var(--risk-low);    border-radius:10px; padding:24px; }
.adv-warn  { background: linear-gradient(135deg,#2d1b00,#4a2d00); border:1px solid var(--risk-medium); border-radius:10px; padding:24px; }
.adv-sell  { background: linear-gradient(135deg,#2d0000,#4a0000); border:1px solid var(--risk-high);   border-radius:10px; padding:24px; }

/* ── Risk Badges ──────────────────────────────────────────────── */
.badge-low    { font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:700; background:rgba(34,201,147,0.08); border:1px solid var(--risk-low);    color:var(--risk-low);    border-radius:4px; padding:2px 8px; letter-spacing:1px; }
.badge-medium { font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:700; background:rgba(232,168,56,0.08);  border:1px solid var(--risk-medium); color:var(--risk-medium); border-radius:4px; padding:2px 8px; letter-spacing:1px; }
.badge-high   { font-family:'JetBrains Mono',monospace; font-size:10px; font-weight:700; background:rgba(232,64,64,0.08);   border:1px solid var(--risk-high);   color:var(--risk-high);   border-radius:4px; padding:2px 8px; letter-spacing:1px; }

/* ── DataFrames ───────────────────────────────────────────────── */
.stDataFrame { border-radius: 8px; overflow: hidden; border: 1px solid var(--border-subtle); }
.stDataFrame td, .stDataFrame th {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important;
}

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--bg-overlay); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.ERROR)

# ─────────────────────────────────────────────────
# Design System constants
# ─────────────────────────────────────────────────
REGIME_COLORS = {
    'Risk-On':      '#22C993',   # green  – risk 15%
    'Defensive':    '#22C993',   # green  – risk 42%
    'Transitional': '#E8A838',   # amber  – risk 65%
    'Crisis':       '#E84040',   # red    – risk 90%
}
CHART_PALETTE = ['#3B82F6','#22C993','#A78BFA','#E8A838','#E84040','#2DD4BF','#FB7185','#86EFAC']

# Shared Plotly layout – all charts extend this
PLOT_BASE = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,28,45,1)',          # --bg-surface
    font=dict(family='JetBrains Mono, Inter', color='#7A9BB5', size=11),
    margin=dict(l=52, r=24, t=36, b=48),
    xaxis=dict(gridcolor='rgba(59,130,246,0.08)', zeroline=False,
               tickfont=dict(size=10, family='JetBrains Mono')),
    yaxis=dict(gridcolor='rgba(59,130,246,0.08)', zeroline=False,
               tickfont=dict(size=10, family='JetBrains Mono')),
    legend=dict(bgcolor='rgba(6,11,20,0.85)',
                bordercolor='rgba(59,130,246,0.20)', borderwidth=1,
                font=dict(size=11, family='JetBrains Mono')),
    hoverlabel=dict(bgcolor='#060B14', bordercolor='#3B82F6',
                    font=dict(family='JetBrains Mono', size=11, color='#E2EAF4')),
)

def _c(**overrides):
    """Return a copy of PLOT_BASE with overrides merged."""
    import copy
    d = copy.deepcopy(PLOT_BASE)
    d.update(overrides)
    return d

def risk_pct(regime: str) -> float:
    return {'Risk-On': 15., 'Defensive': 42., 'Transitional': 65., 'Crisis': 90.}.get(regime, 50.)

def risk_color(p: float) -> str:
    return '#E84040' if p >= 80 else ('#E8A838' if p >= 50 else '#22C993')

def hex_to_rgba(h: str, a: float = 0.15) -> str:
    h = h.lstrip('#')
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

def section(label: str):
    st.markdown(f'<div class="st-section">{label}</div>', unsafe_allow_html=True)

def pct_chart(fig, **kw):
    fig.update_layout(**_c(**kw))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,rgba(59,130,246,0.07) 0%,rgba(13,148,136,0.04) 100%);
            border:1px solid rgba(59,130,246,0.12);border-radius:12px;
            padding:24px 32px;margin-bottom:24px;display:flex;align-items:center;gap:18px">
  <span style="font-size:2.8rem;line-height:1">⚡</span>
  <div>
    <h1 style="font-size:24px;font-weight:800;margin:0;
               background:linear-gradient(90deg,#60a5fa,#34d399 55%,#a78bfa);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               letter-spacing:-0.5px">RegimeEngine</h1>
    <p style="font-size:9px;font-weight:700;text-transform:uppercase;
              letter-spacing:2.5px;color:#3D5A73;margin:4px 0 0">
      Self-Supervised Market Regime Detection &nbsp;·&nbsp;
      TCN Encoder &nbsp;·&nbsp; Gaussian HMM &nbsp;·&nbsp;
      Quant Portfolio Control
    </p>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Configuration")
    TICKERS = [
        "SPY","QQQ","IWM","TLT","GLD","SLV","VNQ",
        "EEM","XLF","XLE","XLK","AAPL","MSFT","TSLA",
        "NVDA","AMZN","META","GOOGL","BTC-USD",
        "SPY,TLT,GLD (Multi-Asset)",
    ]
    tsel = st.selectbox("📈 Ticker / Portfolio", TICKERS, index=0)
    tickers = ([t.strip() for t in tsel.split(",") if t.strip().isalpha()]
               if "," in tsel else [tsel])

    st.markdown("---")
    st.markdown("**📅 Date & Time Range**")
    c1,c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"), key="sd",
                                   label_visibility="collapsed")
        st.caption("Start date")
    with c2:
        start_time = st.time_input("Start Time", dtime(9,30), key="st",
                                   label_visibility="collapsed")
        st.caption("Start time")
    c3,c4 = st.columns(2)
    with c3:
        end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"), key="ed",
                                 label_visibility="collapsed")
        st.caption("End date")
    with c4:
        end_time = st.time_input("End Time", dtime(16,0), key="et",
                                 label_visibility="collapsed")
        st.caption("End time")
    st.caption(f"`{start_date} {start_time}` → `{end_date} {end_time}`")

    st.markdown("---")
    st.markdown("**🔬 Model Hyperparameters**")
    epochs    = st.slider("TCN Epochs",         10, 100, 50, 10)
    n_regimes = st.slider("HMM States",          2,   6,  4)
    window    = st.slider("Lookback Window (d)", 5,  63, 21, 1)

    st.markdown("---")
    st.markdown("**🎲 Monte Carlo Settings**")
    mc_paths   = st.slider("MC Simulation Paths", 500, 10000, 2000, 500)
    mc_horizon = st.slider("Forecast Horizon (d)", 5, 252, 30, 5)

    st.markdown("---")
    run_btn = st.button("🚀 Run RegimeEngine")

# ─────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────
t_over, t_feat, t_tcn, t_hmm, t_port, t_adv, t_abl, t_mc = st.tabs([
    "📊  Overview",
    "🧮  Features",
    "🧠  TCN Encoder",
    "🔬  HMM Regimes",
    "💼  Portfolio",
    "🤖  Advisor",
    "🔬  Ablation Study",
    "🎲  Monte Carlo",
])
# ─────────────────────────────────────────────────
# Welcome screen
# ─────────────────────────────────────────────────
with t_over:
    if not run_btn:
        st.markdown("""
<div style="text-align:center;padding:72px 0 48px">
  <p style="font-size:4rem;margin-bottom:8px">⚡</p>
  <h2 style="font-size:20px;font-weight:700;
             background:linear-gradient(90deg,#60a5fa,#34d399);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;
             margin-bottom:12px">RegimeEngine</h2>
  <p style="color:#3D5A73;font-size:13px;max-width:480px;margin:0 auto 40px;line-height:1.7">
    Select a ticker and date range in the sidebar,<br>
    then click <strong style="color:#3B82F6">🚀 Run RegimeEngine</strong>.
  </p>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;max-width:640px;margin:0 auto">
    <div style="background:#111C2D;border:1px solid rgba(59,130,246,0.10);border-radius:10px;padding:20px;text-align:left">
      <div style="font-size:1.4rem">🧠</div>
      <div style="font-size:11px;font-weight:700;color:#7A9BB5;margin-top:8px;text-transform:uppercase;letter-spacing:1px">TCN Encoder</div>
      <div style="font-size:11px;color:#3D5A73;margin-top:6px;line-height:1.5">Temporal Convolutional Network + Contrastive Self-Supervised Loss</div>
    </div>
    <div style="background:#111C2D;border:1px solid rgba(59,130,246,0.10);border-radius:10px;padding:20px;text-align:left">
      <div style="font-size:1.4rem">🔬</div>
      <div style="font-size:11px;font-weight:700;color:#7A9BB5;margin-top:8px;text-transform:uppercase;letter-spacing:1px">Gaussian HMM</div>
      <div style="font-size:11px;color:#3D5A73;margin-top:6px;line-height:1.5">Latent Space Regime Detection · StandardScaler · 10 EM Restarts</div>
    </div>
    <div style="background:#111C2D;border:1px solid rgba(59,130,246,0.10);border-radius:10px;padding:20px;text-align:left">
      <div style="font-size:1.4rem">💼</div>
      <div style="font-size:11px;font-weight:700;color:#7A9BB5;margin-top:8px;text-transform:uppercase;letter-spacing:1px">Quant Portfolio</div>
      <div style="font-size:11px;color:#3D5A73;margin-top:6px;line-height:1.5">MVO · Min-ES · Hierarchical Risk Parity · Regime-Conditional</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Pipeline (cached)
# ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def run_pipeline_cached(tickers_tuple, start_str, end_str, epochs, n_regimes, window):
    from src.pipeline.main import Pipeline
    p = Pipeline(list(tickers_tuple), start_str, end_str)
    p.tcn_epochs   = epochs
    p.n_components = n_regimes
    p.window_size  = window
    return p.run()

if run_btn:
    with st.status("⚡ RegimeEngine is running…", expanded=True) as sts:
        st.write("📥  Fetching market data & engineering features…")
        try:
            res = run_pipeline_cached(
                tuple(tickers),
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                epochs, n_regimes, window
            )
            st.write("🧠  TCN encoder trained  ·  🔬 HMM fitted  ·  💼 Portfolio allocated")
            sts.update(label="✅  RegimeEngine complete", state="complete")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)
            st.stop()

    prices   = res['prices']
    features = res['features']
    regimes  = res['regimes']
    weights  = res['weights']
    port_val = res['portfolio_value']
    metrics  = res['metrics']
    loss_h   = res.get('loss_history', [])
    latent_z = res.get('latent_z', None)
    hmm_m    = res.get('hmm', None)
    returns  = res.get('returns', None)
    risk_ser = regimes.map(risk_pct)

    # ════════════════════════════════════════
    # TAB 0 — OVERVIEW
    # ════════════════════════════════════════
    with t_over:
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Total Return",      f"{metrics['Total Return']*100:.1f}%")
        m2.metric("Ann. Return",       f"{metrics['Annualized Return']*100:.1f}%")
        m3.metric("Ann. Volatility",   f"{metrics['Annualized Volatility']*100:.1f}%")
        m4.metric("Sharpe Ratio",      f"{metrics['Sharpe Ratio']:.2f}")
        m5.metric("Max Drawdown",      f"{metrics['Max Drawdown']*100:.1f}%")
        cr = regimes.iloc[-1]; crp = risk_pct(cr)
        m6.metric("Current Regime",    cr,
                  delta=f"Risk {crp:.0f}%", delta_color="inverse")

        section("📈  PORTFOLIO VALUE — RISK-SHADED")
        fig_pv = go.Figure()
        aligned_risk = risk_ser.reindex(port_val.index).ffill()
        for lo,hi,col,lbl in [(0,50,'#22C993','Risk < 50%'),
                               (50,80,'#E8A838','Risk 50–80%'),
                               (80,101,'#E84040','Risk > 80%')]:
            mask = (aligned_risk >= lo) & (aligned_risk < hi)
            seg  = port_val[mask]
            if not seg.empty:
                fig_pv.add_trace(go.Scatter(
                    x=seg.index, y=seg.values, mode='lines',
                    line=dict(color=col, width=2.5), name=lbl))
        pct_chart(fig_pv, height=420)

        if returns is not None and len(returns) > 63:
            section("📊  ROLLING 63-DAY SHARPE RATIO")
            eq = returns.mean(axis=1)
            rs = (eq.rolling(63).mean() / eq.rolling(63).std()) * np.sqrt(252)
            fig_rs = go.Figure()
            fig_rs.add_hline(y=0, line=dict(color='rgba(255,255,255,0.15)',
                                            dash='dash', width=1))
            fig_rs.add_trace(go.Scatter(
                x=rs.index, y=rs.values, mode='lines', name='Sharpe (63d)',
                line=dict(color='#A78BFA', width=2),
                fill='tozeroy', fillcolor='rgba(167,139,250,0.06)'))
            pct_chart(fig_rs, height=280)

    # ════════════════════════════════════════
    # TAB 1 — FEATURES
    # ════════════════════════════════════════
    with t_feat:
        fc   = list(features.columns)
        vcols = [c for c in fc if 'volatility' in c]
        rcols = [c for c in fc if 'log_return'  in c]
        pcols = [c for c in fc if 'pos_cusum'   in c]
        ncols = [c for c in fc if 'neg_cusum'   in c]
        ecols = [c for c in fc if 'entropy'     in c]
        xcols = [c for c in fc if 'corr_'       in c]

        section("💹  PRICE SERIES")
        fig_pr = go.Figure()
        for i,col in enumerate(prices.columns[:4]):
            fig_pr.add_trace(go.Scatter(x=prices.index, y=prices[col],
                                         mode='lines', name=col,
                                         line=dict(color=CHART_PALETTE[i], width=1.8)))
        pct_chart(fig_pr, height=320)

        if rcols:
            section("📉  LOG RETURNS")
            fig_lr = go.Figure()
            for i,c in enumerate(rcols[:3]):
                fig_lr.add_trace(go.Scatter(x=features.index, y=features[c],
                                             mode='lines', name=c, opacity=0.85,
                                             line=dict(color=CHART_PALETTE[i], width=1)))
            fig_lr.add_hline(y=0, line=dict(color='rgba(255,255,255,0.12)',
                                             dash='dash', width=1))
            pct_chart(fig_lr, height=260)

        if vcols:
            section("🌊  ROLLING REALIZED VOLATILITY (ANNUALISED)")
            fig_v = go.Figure()
            for i,c in enumerate(vcols[:3]):
                fig_v.add_trace(go.Scatter(
                    x=features.index, y=features[c]*100, mode='lines', name=c,
                    line=dict(color=CHART_PALETTE[i], width=2),
                    fill='tozeroy', fillcolor=hex_to_rgba(CHART_PALETTE[i], 0.06)))
            pct_chart(fig_v, height=280, yaxis=dict(
                ticksuffix='%', gridcolor='rgba(59,130,246,0.08)',
                zeroline=False, tickfont=dict(size=10)))

        if pcols:
            section("🔺  CUSUM STRUCTURAL BREAK DETECTION")
            fig_cs = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.06,
                                    subplot_titles=['Positive CUSUM','Negative CUSUM'])
            for i,c in enumerate(pcols[:2]):
                fig_cs.add_trace(go.Scatter(x=features.index, y=features[c],
                    mode='lines', name=c,
                    line=dict(color='#22C993', width=1.5, dash='solid'
                              if i==0 else 'dot')), row=1, col=1)
            for i,c in enumerate(ncols[:2]):
                fig_cs.add_trace(go.Scatter(x=features.index, y=features[c],
                    mode='lines', name=c,
                    line=dict(color='#E84040', width=1.5, dash='solid'
                              if i==0 else 'dot')), row=2, col=1)
            fig_cs.update_layout(height=380, template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,28,45,1)',
                font=dict(color='#7A9BB5', size=11), showlegend=True,
                margin=dict(l=52, r=24, t=44, b=48))
            st.plotly_chart(fig_cs, use_container_width=True)

        if ecols:
            section("🔐  SHANNON ENTROPY — MARKET UNCERTAINTY")
            fig_e = go.Figure()
            for i,c in enumerate(ecols[:2]):
                fig_e.add_trace(go.Scatter(x=features.index, y=features[c],
                    mode='lines', name=c,
                    line=dict(color=CHART_PALETTE[2+i], width=2)))
            pct_chart(fig_e, height=260)

        if xcols:
            section("🔗  ROLLING CROSS-ASSET CORRELATION")
            fig_xc = go.Figure()
            for i,c in enumerate(xcols[:4]):
                fig_xc.add_trace(go.Scatter(x=features.index, y=features[c],
                    mode='lines', name=c,
                    line=dict(color=CHART_PALETTE[i], width=1.8)))
            fig_xc.add_hline(y=0, line=dict(color='rgba(255,255,255,0.12)',
                                              dash='dash', width=1))
            pct_chart(fig_xc, height=280, yaxis=dict(
                range=[-1.1,1.1], gridcolor='rgba(59,130,246,0.08)',
                zeroline=False, tickfont=dict(size=10)))

        section("🧩  FEATURE CORRELATION MATRIX")
        sf   = features.iloc[:, :min(15, features.shape[1])]
        cmat = sf.corr()
        fig_cm = px.imshow(cmat, color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1, text_auto='.1f', aspect='auto')
        fig_cm.update_layout(height=420, template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#7A9BB5', size=10),
            margin=dict(l=72, r=24, t=36, b=72))
        st.plotly_chart(fig_cm, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 2 — TCN ENCODER
    # ════════════════════════════════════════
    with t_tcn:
        section("📉  CONTRASTIVE TRAINING LOSS")
        if loss_h:
            fig_loss = go.Figure()
            epochs_x = list(range(1, len(loss_h)+1))
            fig_loss.add_trace(go.Scatter(
                x=epochs_x, y=loss_h, mode='lines+markers',
                name='Training Loss',
                line=dict(color='#3B82F6', width=2),
                marker=dict(size=3, color='#93C5FD'),
                fill='tozeroy', fillcolor='rgba(59,130,246,0.06)'))
            if len(loss_h) > 5:
                sm = pd.Series(loss_h).rolling(5, min_periods=1).mean()
                fig_loss.add_trace(go.Scatter(
                    x=epochs_x, y=sm, mode='lines', name='5-epoch avg',
                    line=dict(color='#22C993', width=2, dash='dot')))
            pct_chart(fig_loss, height=320,
                      xaxis=dict(title='Epoch', gridcolor='rgba(59,130,246,0.08)',
                                 zeroline=False),
                      yaxis=dict(title='Loss', gridcolor='rgba(59,130,246,0.08)',
                                 zeroline=False))
        else:
            st.info("Loss history unavailable (served from cache).")

        if latent_z is not None and hmm_m is not None:
            section("🌌  LATENT SPACE — PCA 2D PROJECTION")
            pca   = PCA(n_components=2, random_state=42)
            z2d   = pca.fit_transform(latent_z)
            pred  = hmm_m.predict(latent_z)
            names = [hmm_m.state_map.get(s, f"S{s}") for s in pred]
            zdf   = pd.DataFrame({'PC1':z2d[:,0],'PC2':z2d[:,1],'Regime':names})
            fig_pca = px.scatter(zdf, x='PC1', y='PC2', color='Regime',
                                  color_discrete_map=REGIME_COLORS, opacity=0.6)
            fig_pca.update_traces(marker=dict(size=4))
            pct_chart(fig_pca, height=400)

            section("📊  PCA EXPLAINED VARIANCE")
            ev  = PCA(random_state=42).fit(latent_z).explained_variance_ratio_
            nd  = min(8, len(ev))
            fig_ev = go.Figure(go.Bar(
                x=[f"z_{i+1}" for i in range(nd)],
                y=ev[:nd]*100,
                marker_color='#3B82F6',
                text=[f"{x:.1f}%" for x in ev[:nd]*100],
                textposition='outside', textfont=dict(size=10)))
            pct_chart(fig_ev, height=260,
                      yaxis=dict(ticksuffix='%', gridcolor='rgba(59,130,246,0.08)',
                                 zeroline=False))

            section("📊  LATENT DIMENSION HISTOGRAMS")
            ndim  = min(8, latent_z.shape[1])
            fig_ld = make_subplots(rows=2, cols=4,
                subplot_titles=[f"z_{i+1}" for i in range(ndim)])
            for idx in range(ndim):
                r, c = divmod(idx, 4)
                fig_ld.add_trace(go.Histogram(
                    x=latent_z[:,idx], nbinsx=35,
                    marker_color=CHART_PALETTE[idx], opacity=0.75,
                    showlegend=False), row=r+1, col=c+1)
            fig_ld.update_layout(height=440, template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,28,45,1)',
                font=dict(color='#7A9BB5', size=10),
                margin=dict(l=40, r=20, t=52, b=40))
            st.plotly_chart(fig_ld, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 3 — HMM REGIMES
    # ════════════════════════════════════════
    with t_hmm:
        section("🧭  REGIME-SHADED PRICE CHART")
        first_tick = prices.columns[0]
        pser = prices[first_tick].reindex(regimes.index).ffill()
        fig_reg = go.Figure()
        prev_r, seg_s = regimes.iloc[0], regimes.index[0]
        for dt, r in regimes.items():
            if r != prev_r:
                fig_reg.add_vrect(x0=seg_s, x1=dt,
                    fillcolor=REGIME_COLORS.get(prev_r,'#3B82F6'),
                    opacity=0.10, line_width=0)
                seg_s = dt; prev_r = r
        fig_reg.add_vrect(x0=seg_s, x1=regimes.index[-1],
            fillcolor=REGIME_COLORS.get(prev_r,'#3B82F6'),
            opacity=0.10, line_width=0)
        fig_reg.add_trace(go.Scatter(
            x=pser.index, y=pser.values, mode='lines', name=first_tick,
            line=dict(color='#E2EAF4', width=1.5)))
        for nm, cl in REGIME_COLORS.items():
            fig_reg.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                marker=dict(size=9, color=cl), name=nm))
        pct_chart(fig_reg, height=440,
                  xaxis=dict(rangeslider=dict(visible=True, thickness=0.05),
                             gridcolor='rgba(59,130,246,0.08)', zeroline=False))

        section("🌡  RISK LEVEL OVER TIME")
        fig_rk = go.Figure()
        for y0,y1,col,lbl in [(0,50,'#22C993','Low'),
                               (50,80,'#E8A838','Medium'),
                               (80,100,'#E84040','High')]:
            fig_rk.add_hrect(y0=y0, y1=y1, fillcolor=col, opacity=0.04,
                              line_width=0,
                              annotation_text=f'{lbl} Risk',
                              annotation_position='top left',
                              annotation_font=dict(color=col, size=9))
        fig_rk.add_hline(y=50, line=dict(color='#E8A838', dash='dot', width=1))
        fig_rk.add_hline(y=80, line=dict(color='#E84040', dash='dot', width=1))
        fig_rk.add_trace(go.Scatter(
            x=risk_ser.index, y=risk_ser.values, mode='lines', name='Risk %',
            fill='tozeroy', fillcolor='rgba(59,130,246,0.05)',
            line=dict(color='#3B82F6', width=2)))
        pct_chart(fig_rk, height=280,
                  yaxis=dict(range=[0,100], ticksuffix='%',
                             gridcolor='rgba(59,130,246,0.08)', zeroline=False))

        if hmm_m is not None:
            section("🔀  HMM STATE TRANSITION MATRIX")
            transmat = hmm_m.get_transition_matrix()
            rlabels  = [hmm_m.state_map.get(i,f"S{i}") for i in range(len(transmat))]
            fig_tm = px.imshow(transmat, x=rlabels, y=rlabels,
                color_continuous_scale='Blues', text_auto='.2f',
                labels=dict(x='To Regime', y='From Regime', color='Prob'))
            fig_tm.update_layout(height=380, template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#7A9BB5', size=11),
                margin=dict(l=80, r=24, t=36, b=80))
            st.plotly_chart(fig_tm, use_container_width=True)

            section("⏱  REGIME PERSISTENCE")
            cols_p = st.columns(len(transmat))
            for i,(lbl,col) in enumerate(zip(rlabels,cols_p)):
                diag = transmat[i][i]
                per  = 1/(1-diag) if diag < 1 else float('inf')
                col.metric(lbl, f"{per:.1f}d", delta=f"P(stay)={diag:.2f}")

            # Frequency bar chart
            section("📊  REGIME FREQUENCY (DAYS)")
            pred_states = hmm_m.predict(latent_z)
            pred_names  = [hmm_m.state_map.get(s,f"S{s}") for s in pred_states]
            rc = pd.Series(pred_names).value_counts()
            fig_rc = go.Figure(go.Bar(
                x=rc.index, y=rc.values,
                marker_color=[REGIME_COLORS.get(r,'#3B82F6') for r in rc.index],
                text=rc.values, textposition='outside',
                textfont=dict(family='JetBrains Mono', size=11)))
            pct_chart(fig_rc, height=280,
                      xaxis=dict(gridcolor='rgba(59,130,246,0.08)', zeroline=False),
                      yaxis=dict(title='Days', gridcolor='rgba(59,130,246,0.08)',
                                 zeroline=False))

        if returns is not None:
            section("🎻  RETURN DISTRIBUTION PER REGIME")
            merged = pd.concat([returns.mean(axis=1).rename('ret'),
                                 regimes.rename('regime')],axis=1).dropna()
            fig_vp = go.Figure()
            for rname in merged['regime'].unique():
                sub = merged[merged['regime']==rname]['ret'] * 100
                col = REGIME_COLORS.get(rname,'#3B82F6')
                fig_vp.add_trace(go.Violin(
                    y=sub, name=rname, line_color=col,
                    box_visible=True, meanline_visible=True,
                    fillcolor=hex_to_rgba(col, 0.12)))
            pct_chart(fig_vp, height=360,
                      yaxis=dict(title='Daily Return (%)',
                                 gridcolor='rgba(59,130,246,0.08)',
                                 zeroline=False))

    # ════════════════════════════════════════
    # TAB 4 — PORTFOLIO
    # ════════════════════════════════════════
    with t_port:
        section("📋  RECENT ALLOCATIONS + RISK LEVEL")
        rec = weights.tail(20).copy()
        rec.insert(0, 'Risk %', regimes.reindex(rec.index).ffill().map(risk_pct).values.round(1))
        rec.insert(0, 'Regime', regimes.reindex(rec.index).ffill().values)

        def hi_r(v):
            if isinstance(v, float) and 0 <= v <= 100:
                if v >= 80: return 'background-color:rgba(232,64,64,0.18);color:#E84040;font-weight:700'
                if v >= 50: return 'background-color:rgba(232,168,56,0.18);color:#E8A838;font-weight:700'
                return 'background-color:rgba(34,201,147,0.18);color:#22C993;font-weight:700'
            return ''
        styled = rec.style.applymap(hi_r, subset=['Risk %'])
        for c in weights.columns:
            styled = styled.format({c:'{:.1%}'}, na_rep='-')
        st.dataframe(styled, use_container_width=True, height=400)

        section("🗂  ALLOCATION OVER TIME (STACKED AREA)")
        wk = weights.resample('W').mean().fillna(0)
        fig_wa = go.Figure()
        for i,c in enumerate(wk.columns):
            fig_wa.add_trace(go.Scatter(
                x=wk.index, y=wk[c]*100, mode='lines', name=c,
                stackgroup='one',
                line=dict(color=CHART_PALETTE[i%len(CHART_PALETTE)], width=0.5),
                fillcolor=hex_to_rgba(CHART_PALETTE[i%len(CHART_PALETTE)], 0.70)))
        pct_chart(fig_wa, height=320,
                  yaxis=dict(ticksuffix='%', gridcolor='rgba(59,130,246,0.08)',
                             zeroline=False))

        section("📉  DRAWDOWN (UNDERWATER EQUITY CURVE)")
        peak = port_val.cummax()
        dd   = (port_val - peak) / peak * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values, mode='lines', name='Drawdown',
            fill='tozeroy', fillcolor='rgba(232,64,64,0.10)',
            line=dict(color='#E84040', width=1.8)))
        pct_chart(fig_dd, height=260,
                  yaxis=dict(ticksuffix='%', gridcolor='rgba(59,130,246,0.08)',
                             zeroline=False))

        if returns is not None:
            section("📈  PER-REGIME STATISTICS")
            eq_ret  = returns.mean(axis=1)
            reg_ali = regimes.reindex(eq_ret.index).ffill()
            rows = []
            for rn in reg_ali.unique():
                rs = eq_ret[reg_ali==rn]
                rows.append({'Regime':rn,
                    'Ann. Return': f"{rs.mean()*252*100:.1f}%",
                    'Ann. Vol':    f"{rs.std()*np.sqrt(252)*100:.1f}%",
                    'Sharpe':      f"{(rs.mean()*252)/(rs.std()*np.sqrt(252)+1e-9):.2f}",
                    'Max DD':      f"{((rs+1).cumprod()/((rs+1).cumprod().cummax())-1).min()*100:.1f}%",
                    'Days':        len(rs)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

    # ════════════════════════════════════════
    # TAB 5 — ADVISOR
    # ════════════════════════════════════════
    with t_adv:
        cur   = regimes.iloc[-1]
        crp   = risk_pct(cur)
        rt    = risk_ser.tail(10).diff().mean()

        section("📡  CURRENT MARKET PULSE")
        a1,a2,a3 = st.columns(3)
        a1.metric("Active Regime",    cur)
        a2.metric("Risk Score",       f"{crp:.0f}%",
                  delta=f"{rt:+.2f}% / day")
        a3.metric("Risk Status",
                  "HIGH" if crp>=80 else ("MEDIUM" if crp>=50 else "LOW"),
                  delta="↑ Increasing" if rt>0.5 else ("↓ Improving" if rt<-0.5 else "→ Stable"),
                  delta_color="inverse")

        st.markdown("---")

        if crp < 50:
            cls, icon, action = 'adv-buy', '📈', 'BUY / ADD RISK'
            body = f"""**Regime: {cur}  ·  Risk {crp:.0f}% — Low Zone**

RegimeEngine detects **bullish, low-volatility** conditions consistent with a Risk-On regime.

| Action | Detail |
|---|---|
| ✅ Equities | Overweight — growth / momentum sectors |
| ✅ Allocation | MVO-optimised; maximize return per unit vol |
| ✅ Leverage | 1.1–1.25× portfolio-level acceptable |
| ⚠ Bonds | Underweight; rotate into equities |

*Stop-loss: 5–7% below entry · Monitor 50% risk threshold daily.*"""
        elif crp < 80:
            cls, icon, action = 'adv-warn', '⚠', 'HOLD / TRIM RISK'
            body = f"""**Regime: {cur}  ·  Risk {crp:.0f}% — Medium Zone**

Conditions are **elevated** — RegimeEngine detects Transitional/Defensive characteristics.

| Action | Detail |
|---|---|
| ⚠ Equities  | Trim high-beta by 15–20%; prefer low-vol |
| ✅ Bonds     | Add TLT / short-duration credit |
| ✅ Gold      | Maintain 5–10% as regime hedge |
| ⚠ Allocation | Shift to ES-minimising weights |

*Tighten stops · 3-day confirmation before full de-risking.*"""
        else:
            cls, icon, action = 'adv-sell', '🔴', 'REDUCE / HEDGE'
            body = f"""**Regime: {cur}  ·  Risk {crp:.0f}% — HIGH RISK ZONE**

RegimeEngine is in **CRISIS mode.** Historical drawdowns are extreme.

| Action | Detail |
|---|---|
| 🔴 Equities  | Reduce to < 25% max exposure |
| 🔴 Hedges   | Add VIX calls, SPY put spreads |
| ✅ Safe havens | TLT, GLD, USD cash — target 60–75% |
| ✅ Allocation | HRP weights for remaining equity |

*Circuit breaker: 10% portfolio peak-to-trough = full exit.*"""

        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        st.markdown(f"## {icon} {action}")
        st.markdown(body)
        st.markdown('</div>', unsafe_allow_html=True)

        section("📊  HISTORICAL RISK PROFILE")
        fig_adv = go.Figure()
        for y0,y1,col,lbl in [(0,50,'#22C993','Low'),
                               (50,80,'#E8A838','Medium'),
                               (80,100,'#E84040','High')]:
            fig_adv.add_hrect(y0=y0, y1=y1, fillcolor=col, opacity=0.04,
                               line_width=0,
                               annotation_text=f'{lbl} Zone',
                               annotation_position='top left',
                               annotation_font=dict(color=col, size=9))
        fig_adv.add_hline(y=50, line=dict(color='#E8A838', dash='dot', width=1))
        fig_adv.add_hline(y=80, line=dict(color='#E84040', dash='dot', width=1))
        fig_adv.add_trace(go.Scatter(
            x=risk_ser.index, y=risk_ser.values, mode='lines',
            fill='tozeroy', fillcolor='rgba(59,130,246,0.05)',
            line=dict(color='#3B82F6', width=2), name='Risk %'))
        fig_adv.add_hline(y=crp,
            line=dict(color='#E2EAF4', dash='dash', width=1.5),
            annotation_text=f'Now: {crp:.0f}%',
            annotation_font=dict(color='#E2EAF4', size=10))
        pct_chart(fig_adv, height=300,
                  yaxis=dict(range=[0,100], ticksuffix='%',
                             gridcolor='rgba(59,130,246,0.08)', zeroline=False))

# ════════════════════════════════════════
# TAB 6 — ABLATION STUDY
# ════════════════════════════════════════
with t_abl:
    section("🔬  ABLATION STUDY — REGIME DETECTION METHOD COMPARISON")
    st.markdown("""
<div style="background:#111C2D;border:1px solid rgba(59,130,246,0.10);border-radius:10px;
            padding:18px 24px;margin-bottom:20px;font-size:12px;color:#7A9BB5;line-height:1.7">
<strong style="color:#E2EAF4">Study Design</strong> — Four methods are compared on identical
data using five unsupervised quality metrics to isolate the contribution of the TCN encoder.<br><br>
<span style="color:#3B82F6;font-weight:700">M0</span> Raw Returns HMM &nbsp;·&nbsp;
<span style="color:#A78BFA;font-weight:700">M1</span> Engineered Features HMM &nbsp;·&nbsp;
<span style="color:#E8A838;font-weight:700">M2</span> PCA(8d) + HMM (classical) &nbsp;·&nbsp;
<span style="color:#22C993;font-weight:700">M3</span> Contrastive TCN + HMM <em>(Ours)</em>
</div>""", unsafe_allow_html=True)

    _has_data = 'latent_z' in dir() and latent_z is not None and 'returns' in dir() and returns is not None
    if _has_data:
        with st.spinner("⚡ Running ablation (4 methods × 5 metrics)…"):
            try:
                from src.evaluation.ablation import run_ablation
                from sklearn.preprocessing import StandardScaler as _SS2
                abl_results, abl_summary = run_ablation(
                    returns_df=returns,
                    features_df=features,
                    latent_z=latent_z,
                    n_components=n_regimes,
                    window_size=window,
                )
            except Exception as _e:
                st.error(f"Ablation error: {_e}")
                st.exception(_e)
                abl_results, abl_summary = None, None

        if abl_results is not None:
            M_COLS = ["#3B82F6", "#A78BFA", "#E8A838", "#22C993"]
            mnames = [r.name for r in abl_results]

            # ── 5 Metric Bar Charts ───────────────────────────────────
            section("📊  FIVE-METRIC COMPARISON (🟢 = best per metric)")
            metric_defs = [
                ("Silhouette ↑",           [r.silhouette    for r in abl_results], True),
                ("Calinski-Harabasz ↑",    [r.ch_score      for r in abl_results], True),
                ("Davies-Bouldin ↓",       [r.db_score      for r in abl_results], False),
                ("Stability (days) ↑",     [r.stability     for r in abl_results], True),
                ("Pred. Validity (ρ) ↑",   [r.pred_validity for r in abl_results], True),
            ]
            cols5 = st.columns(5)
            for ci, (mname, vals, hb) in enumerate(metric_defs):
                best = max(vals) if hb else min(vals)
                bar_cols = ["#22C993" if v == best else c for v, c in zip(vals, M_COLS)]
                fig_mb = go.Figure(go.Bar(
                    x=[n.split(":")[0] for n in mnames],
                    y=vals,
                    marker_color=bar_cols,
                    text=[f"{v:.3f}" for v in vals],
                    textposition="outside",
                    textfont=dict(size=9, family="JetBrains Mono"),
                ))
                fig_mb.update_layout(
                    height=250,
                    title=dict(text=mname, font=dict(size=10, color="#7A9BB5")),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(17,28,45,1)",
                    font=dict(color="#7A9BB5", size=9),
                    margin=dict(l=30, r=10, t=44, b=36),
                    xaxis=dict(gridcolor="rgba(59,130,246,0.08)", zeroline=False),
                    yaxis=dict(gridcolor="rgba(59,130,246,0.08)", zeroline=False),
                )
                cols5[ci].plotly_chart(fig_mb, use_container_width=True)

            # ── PCA 2D scatter per method ─────────────────────────────
            section("🌌  LATENT SPACE STRUCTURE — PCA 2D PROJECTION")
            fig_sc = make_subplots(
                rows=2, cols=2,
                subplot_titles=[r.name for r in abl_results],
                horizontal_spacing=0.08, vertical_spacing=0.14,
            )
            for mi, r in enumerate(abl_results):
                grow, gcol = divmod(mi, 2)
                try:
                    sp = r.space
                    if sp.shape[1] > 2:
                        from sklearn.preprocessing import StandardScaler as _SS3
                        sp = PCA(n_components=2, random_state=42).fit_transform(
                            _SS3().fit_transform(sp))
                    elif sp.shape[1] == 1:
                        sp = np.hstack([sp, np.zeros_like(sp)])
                    for lbl in np.unique(r.labels):
                        msk = r.labels == lbl
                        fig_sc.add_trace(
                            go.Scatter(
                                x=sp[msk, 0], y=sp[msk, 1],
                                mode="markers",
                                marker=dict(size=3, opacity=0.5,
                                            color=CHART_PALETTE[int(lbl) % len(CHART_PALETTE)]),
                                name=f"S{lbl}",
                                showlegend=(mi == 0),
                            ),
                            row=grow + 1, col=gcol + 1,
                        )
                except Exception:
                    pass
            fig_sc.update_layout(
                height=520, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,28,45,1)",
                font=dict(color="#7A9BB5", size=10),
                margin=dict(l=40, r=20, t=60, b=40),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            # ── Ranked Summary Table ──────────────────────────────────
            section("🏆  OVERALL RANKING SUMMARY")
            disp = abl_summary.drop(columns=["Notes"], errors="ignore").reset_index()

            def _style_rank(v):
                if v == 1:
                    return "background-color:rgba(34,201,147,0.18);color:#22C993;font-weight:700"
                if v == 2:
                    return "background-color:rgba(59,130,246,0.12);color:#60a5fa"
                return ""

            st.dataframe(
                disp.style.applymap(_style_rank, subset=["Overall Rank"]),
                use_container_width=True, hide_index=True, height=192,
            )

            wi  = abl_summary["Overall Rank"].idxmin()
            wri = int(abl_summary["Overall Rank"].values.argmin())
            wr  = abl_results[wri]
            st.markdown(f"""
<div style="background:linear-gradient(135deg,#012e1f,#024d33);
            border:1px solid #22C993;border-radius:10px;
            padding:18px 24px;margin-top:12px">
  <strong style="color:#22C993;font-size:14px">🏆 Winning Method: {wi}</strong><br>
  <span style="color:#7A9BB5;font-size:11px;font-family:'JetBrains Mono',monospace">
    Silhouette={wr.silhouette:.4f} &nbsp; CH={wr.ch_score:.1f} &nbsp;
    DB={wr.db_score:.4f} &nbsp; Stability={wr.stability:.1f}d &nbsp;
    PredVal=\u03c1{wr.pred_validity:.4f}
  </span>
</div>""", unsafe_allow_html=True)

            # ── Metric explanations ───────────────────────────────────
            with st.expander("ℹ️  Metric Definitions (click to expand)"):
                st.markdown("""
| Metric | Direction | Interpretation |
|---|---|---|
| **Silhouette** | ↑ Higher | How compact and separated the regime clusters are. Range [−1, +1]; >0.5 = strong structure. |
| **Calinski-Harabasz** | ↑ Higher | Ratio of between-cluster to within-cluster dispersion. No upper bound. |
| **Davies-Bouldin** | ↓ Lower | Average similarity of each cluster to its most similar cluster. DB=0 is perfect. |
| **Regime Stability** | ↑ Higher | Mean consecutive days in same regime. Regimes < 3 days are not tradeable in practice. Target ≥ 10d. |
| **Predictive Validity** | ↑ Higher | Spearman ρ between regime label and next-21d realized volatility. Tests economic meaning. |
""")
    else:
        st.info("▶  Run the pipeline first (🚀 Run RegimeEngine) to enable the ablation study.")

# ════════════════════════════════════════
# TAB 7 — MONTE CARLO SIMULATION
# ════════════════════════════════════════
with t_mc:
    section("🎲  ADVANCED MONTE CARLO SIMULATION")
    st.markdown("""
<div style="background:#111C2D;border:1px solid rgba(59,130,246,0.10);border-radius:10px;
            padding:18px 24px;margin-bottom:20px;font-size:12px;color:#7A9BB5;line-height:1.7">
<strong style="color:#E2EAF4">Regime-Switching Monte Carlo Engine</strong> — Forward-looking simulation
using the HMM's learned transition matrix and regime-specific return distributions.<br><br>
<strong>GBM Formula:</strong>
<code style="color:#A78BFA">P(t+1) = P(t) × exp((μ − 0.5σ²)·Δt + σ·√Δt·Z)</code>
where Z ~ N(0,1)<br>
Paths: <code>{mc_paths}</code> · Horizon: <code>{mc_horizon}d</code>
</div>""".format(mc_paths=mc_paths, mc_horizon=mc_horizon), unsafe_allow_html=True)

    _mc_ready = ('hmm_m' in dir() and hmm_m is not None
                 and 'latent_z' in dir() and latent_z is not None
                 and 'returns' in dir() and returns is not None
                 and 'prices' in dir() and prices is not None)

    if _mc_ready:
        from src.backtest.monte_carlo import RegimeSwitchingMonteCarlo, GBMSimulator

        # ── Derive params from pipeline data ──────────────────────────
        first_tick   = prices.columns[0]
        start_price  = float(prices[first_tick].iloc[-1])
        eq_ret       = returns.mean(axis=1)
        mu_daily     = float(eq_ret.mean())
        sigma_daily  = float(eq_ret.std())

        # Current regime and transition matrix
        cur_regime_name = regimes.iloc[-1]
        transmat        = hmm_m.get_transition_matrix()
        pred_states     = hmm_m.predict(latent_z)
        n_states        = hmm_m.n_components

        # Build per-regime mu/cov from observed returns
        regime_params = {}
        for s in range(n_states):
            mask = pred_states == s
            rets_in_s = returns.values[mask] if mask.any() else returns.values[:10]
            regime_params[s] = {
                'mu':  rets_in_s.mean(axis=0),
                'cov': np.cov(rets_in_s.T) + np.eye(rets_in_s.shape[1]) * 1e-6,
            }

        # Find integer state for current regime
        cur_state = 0
        for s, name in hmm_m.state_map.items():
            if name == cur_regime_name:
                cur_state = s
                break

        with st.spinner("⚡ Running Monte Carlo simulations…"):
            # ── 1) Regime-Switching MC ─────────────────────────────────
            mc_engine = RegimeSwitchingMonteCarlo(transmat, regime_params)
            mc_res    = mc_engine.simulate(cur_state, mc_horizon, n_paths=mc_paths)

            # Equal-weight portfolio paths
            n_assets   = mc_res['returns'].shape[2]
            eq_weights = np.ones(n_assets) / n_assets
            port_paths = mc_engine.simulate_portfolio_paths(
                mc_res['returns'], eq_weights, initial_value=start_price)

            # ── 2) Standard GBM for comparison ────────────────────────
            gbm_paths = GBMSimulator.simulate_gbm(
                start_price, mu_daily, sigma_daily,
                mc_horizon, mc_paths, seed=42)

        # ══════════════════════════════════════════
        # SECTION 1 — REGIME-SWITCHING FAN CHART
        # ══════════════════════════════════════════
        section("📈  REGIME-SWITCHING FORWARD SIMULATION")
        rs_stats = GBMSimulator.fan_chart_stats(port_paths)
        rs_probs = GBMSimulator.compute_probabilities(port_paths, start_price)
        rs_risk  = GBMSimulator.compute_risk_metrics(port_paths, start_price)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Expected Price", f"${rs_probs['expected']:,.2f}",
                   delta=f"{(rs_probs['expected']/start_price - 1)*100:+.2f}%")
        mc2.metric("P(Gain)",        f"{rs_probs['P(gain)']:.1f}%")
        mc3.metric("VaR 95%",        f"{rs_risk['VaR_95']:.2f}%")
        mc4.metric("CVaR 95%",       f"{rs_risk['CVaR_95']:.2f}%")

        days_x = list(range(mc_horizon))
        fig_fan = go.Figure()
        # P5-P95 band
        fig_fan.add_trace(go.Scatter(
            x=days_x + days_x[::-1],
            y=list(rs_stats['p95']) + list(rs_stats['p5'][::-1]),
            fill='toself', fillcolor='rgba(59,130,246,0.08)',
            line=dict(width=0), name='P5–P95 Band', showlegend=True))
        # P25-P75 band
        fig_fan.add_trace(go.Scatter(
            x=days_x + days_x[::-1],
            y=list(rs_stats['p75']) + list(rs_stats['p25'][::-1]),
            fill='toself', fillcolor='rgba(59,130,246,0.18)',
            line=dict(width=0), name='P25–P75 Band', showlegend=True))
        # Median & mean
        fig_fan.add_trace(go.Scatter(
            x=days_x, y=rs_stats['median'], mode='lines',
            name='Median', line=dict(color='#00D4FF', width=3)))
        fig_fan.add_trace(go.Scatter(
            x=days_x, y=rs_stats['mean'], mode='lines',
            name='Mean', line=dict(color='#A78BFA', width=2, dash='dot')))
        # P5 and P95 lines
        fig_fan.add_trace(go.Scatter(
            x=days_x, y=rs_stats['p5'], mode='lines',
            name='P5', line=dict(color='#E84040', width=1.5, dash='dash')))
        fig_fan.add_trace(go.Scatter(
            x=days_x, y=rs_stats['p95'], mode='lines',
            name='P95', line=dict(color='#22C993', width=1.5, dash='dash')))
        # Current price line
        fig_fan.add_hline(y=start_price,
            line=dict(color='rgba(255,255,255,0.4)', dash='dot', width=1),
            annotation_text=f'Current: ${start_price:,.0f}',
            annotation_font=dict(color='#E2EAF4', size=10))
        pct_chart(fig_fan, height=420,
                  xaxis=dict(title='Trading Day', gridcolor='rgba(59,130,246,0.08)'),
                  yaxis=dict(title='Price ($)', gridcolor='rgba(59,130,246,0.08)'))

        # ══════════════════════════════════════════
        # SECTION 2 — 3D MONTE CARLO TORNADO
        # ══════════════════════════════════════════
        section("🌪  3D MONTE CARLO PATH TORNADO")
        st.caption("Drag to rotate · Scroll to zoom · Green = gain, Red = loss")

        n_sample = min(300, mc_paths)
        sample_idx = np.random.choice(mc_paths, n_sample, replace=False)
        fig_3d = go.Figure()

        for si, idx in enumerate(sample_idx):
            path = gbm_paths[idx]
            final_gain = path[-1] > start_price
            color = 'rgba(34,201,147,0.07)' if final_gain else 'rgba(232,64,64,0.07)'
            fig_3d.add_trace(go.Scatter3d(
                x=list(range(len(path))), y=[si]*len(path), z=path.tolist(),
                mode='lines', line=dict(color=color, width=1),
                showlegend=False, hoverinfo='skip'))

        # Mean path overlay
        gbm_stats = GBMSimulator.fan_chart_stats(gbm_paths)
        mid_y = n_sample // 2
        fig_3d.add_trace(go.Scatter3d(
            x=list(range(len(gbm_stats['mean']))),
            y=[mid_y]*len(gbm_stats['mean']),
            z=gbm_stats['mean'].tolist(),
            mode='lines', line=dict(color='#00D4FF', width=6),
            name='Mean Path'))
        # P5 path
        fig_3d.add_trace(go.Scatter3d(
            x=list(range(len(gbm_stats['p5']))),
            y=[10]*len(gbm_stats['p5']),
            z=gbm_stats['p5'].tolist(),
            mode='lines', line=dict(color='#E84040', width=3, dash='dash'),
            name='P5'))
        # P95 path
        fig_3d.add_trace(go.Scatter3d(
            x=list(range(len(gbm_stats['p95']))),
            y=[n_sample - 10]*len(gbm_stats['p95']),
            z=gbm_stats['p95'].tolist(),
            mode='lines', line=dict(color='#22C993', width=3, dash='dash'),
            name='P95'))

        fig_3d.update_layout(
            height=560, template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                bgcolor='#030610',
                xaxis=dict(title='Trading Day', gridcolor='rgba(59,130,246,0.12)',
                           color='#3D5A73'),
                yaxis=dict(title='Simulation #', gridcolor='rgba(59,130,246,0.12)',
                           color='#3D5A73'),
                zaxis=dict(title='Price ($)', gridcolor='rgba(59,130,246,0.12)',
                           color='#7A9BB5'),
                camera=dict(eye=dict(x=1.8, y=1.2, z=0.9)),
            ),
            font=dict(color='#7A9BB5', size=10),
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # ══════════════════════════════════════════
        # SECTION 3 — 2D DENSITY HEATMAP
        # ══════════════════════════════════════════
        section("🔥  PRICE-TIME DENSITY HEATMAP")

        density, price_mids, days_arr = GBMSimulator.density_matrix(gbm_paths, n_price_bins=50)

        fig_hm = go.Figure()
        fig_hm.add_trace(go.Heatmap(
            z=density, x=days_arr, y=price_mids,
            colorscale=[
                [0, '#030610'], [0.2, '#003060'], [0.5, '#006090'],
                [0.75, '#00C0D0'], [0.9, '#00FF88'], [1.0, '#FFFF00']],
            showscale=True, colorbar=dict(
                title='Density', titlefont=dict(color='#7A9BB5', size=10),
                tickfont=dict(color='#7A9BB5', size=9)),
            hovertemplate='Day %{x}<br>Price $%{y:,.0f}<br>Density %{z:.4f}<extra></extra>'))
        # Overlay mean path
        fig_hm.add_trace(go.Scatter(
            x=days_arr, y=gbm_stats['mean'][:-1] if len(gbm_stats['mean']) > len(days_arr) else gbm_stats['mean'][:len(days_arr)],
            mode='lines', name='Mean',
            line=dict(color='#00D4FF', width=2.5)))
        fig_hm.add_trace(go.Scatter(
            x=days_arr, y=gbm_stats['p5'][:-1] if len(gbm_stats['p5']) > len(days_arr) else gbm_stats['p5'][:len(days_arr)],
            mode='lines', name='P5',
            line=dict(color='#E84040', width=1.5, dash='dash')))
        fig_hm.add_trace(go.Scatter(
            x=days_arr, y=gbm_stats['p95'][:-1] if len(gbm_stats['p95']) > len(days_arr) else gbm_stats['p95'][:len(days_arr)],
            mode='lines', name='P95',
            line=dict(color='#22C993', width=1.5, dash='dash')))
        fig_hm.add_hline(y=start_price,
            line=dict(color='rgba(255,255,255,0.5)', dash='dot', width=1))
        pct_chart(fig_hm, height=420,
                  xaxis=dict(title='Trading Day', gridcolor='rgba(59,130,246,0.08)'),
                  yaxis=dict(title='Price ($)', gridcolor='rgba(59,130,246,0.08)'))

        # ══════════════════════════════════════════
        # SECTION 4 — PROBABILITY DASHBOARD
        # ══════════════════════════════════════════
        section("📊  PROBABILITY DASHBOARD")

        gbm_probs = GBMSimulator.compute_probabilities(gbm_paths, start_price)
        gbm_risk  = GBMSimulator.compute_risk_metrics(gbm_paths, start_price)

        # Bullish probabilities
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.markdown(f"""
<div style="background:rgba(34,201,147,0.08);border:1px solid #22C993;border-radius:10px;padding:16px;text-align:center">
  <div style="font-size:10px;color:#3D5A73;text-transform:uppercase;letter-spacing:1.5px">P(Gain)</div>
  <div style="font-size:28px;font-weight:800;color:#22C993;font-family:'JetBrains Mono'">{gbm_probs['P(gain)']:.1f}%</div>
</div>""", unsafe_allow_html=True)
        bc2.markdown(f"""
<div style="background:rgba(34,201,147,0.08);border:1px solid #22C993;border-radius:10px;padding:16px;text-align:center">
  <div style="font-size:10px;color:#3D5A73;text-transform:uppercase;letter-spacing:1.5px">P(+2%)</div>
  <div style="font-size:28px;font-weight:800;color:#22C993;font-family:'JetBrains Mono'">{gbm_probs['P(+2%)']:.1f}%</div>
</div>""", unsafe_allow_html=True)
        bc3.markdown(f"""
<div style="background:rgba(34,201,147,0.08);border:1px solid #22C993;border-radius:10px;padding:16px;text-align:center">
  <div style="font-size:10px;color:#3D5A73;text-transform:uppercase;letter-spacing:1.5px">P(+5%)</div>
  <div style="font-size:28px;font-weight:800;color:#22C993;font-family:'JetBrains Mono'">{gbm_probs['P(+5%)']:.1f}%</div>
</div>""", unsafe_allow_html=True)
        bc4.markdown(f"""
<div style="background:rgba(34,201,147,0.08);border:1px solid #22C993;border-radius:10px;padding:16px;text-align:center">
  <div style="font-size:10px;color:#3D5A73;text-transform:uppercase;letter-spacing:1.5px">P(+10%)</div>
  <div style="font-size:28px;font-weight:800;color:#22C993;font-family:'JetBrains Mono'">{gbm_probs['P(+10%)']:.1f}%</div>
</div>""", unsafe_allow_html=True)

        # Bearish probabilities
        br1, br2, br3, br4 = st.columns(4)
        br1.markdown(f"""
<div style="background:rgba(232,64,64,0.08);border:1px solid #E84040;border-radius:10px;padding:16px;text-align:center">
  <div style="font-size:10px;color:#3D5A73;text-transform:uppercase;letter-spacing:1.5px">P(Loss)</div>
  <div style="font-size:28px;font-weight:800;color:#E84040;font-family:'JetBrains Mono'">{gbm_probs['P(loss)']:.1f}%</div>
</div>""", unsafe_allow_html=True)
        br2.markdown(f"""
<div style="background:rgba(232,64,64,0.08);border:1px solid #E84040;border-radius:10px;padding:16px;text-align:center">
  <div style="font-size:10px;color:#3D5A73;text-transform:uppercase;letter-spacing:1.5px">P(−2%)</div>
  <div style="font-size:28px;font-weight:800;color:#E84040;font-family:'JetBrains Mono'">{gbm_probs['P(-2%)']:.1f}%</div>
</div>""", unsafe_allow_html=True)
        br3.markdown(f"""
<div style="background:rgba(232,64,64,0.08);border:1px solid #E84040;border-radius:10px;padding:16px;text-align:center">
  <div style="font-size:10px;color:#3D5A73;text-transform:uppercase;letter-spacing:1.5px">P(−5%)</div>
  <div style="font-size:28px;font-weight:800;color:#E84040;font-family:'JetBrains Mono'">{gbm_probs['P(-5%)']:.1f}%</div>
</div>""", unsafe_allow_html=True)
        br4.markdown(f"""
<div style="background:rgba(232,64,64,0.08);border:1px solid #E84040;border-radius:10px;padding:16px;text-align:center">
  <div style="font-size:10px;color:#3D5A73;text-transform:uppercase;letter-spacing:1.5px">P(−10%)</div>
  <div style="font-size:28px;font-weight:800;color:#E84040;font-family:'JetBrains Mono'">{gbm_probs['P(-10%)']:.1f}%</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk metrics
        rk1, rk2, rk3, rk4 = st.columns(4)
        rk1.metric("VaR 95% (GBM)", f"{gbm_risk['VaR_95']:.2f}%")
        rk2.metric("CVaR 95% (GBM)", f"{gbm_risk['CVaR_95']:.2f}%")
        rk3.metric("VaR Price", f"${gbm_risk['VaR_price']:,.2f}")
        rk4.metric("CVaR Price", f"${gbm_risk['CVaR_price']:,.2f}")

        # Comparison table
        st.markdown("<br>", unsafe_allow_html=True)
        section("⚖  GBM vs REGIME-SWITCHING COMPARISON")
        comp_df = pd.DataFrame({
            'Metric': ['Expected Price', 'Median Price', 'P(Gain)', 'VaR 95%', 'CVaR 95%',
                       'P(+5%)', 'P(−5%)'],
            'GBM Model': [
                f"${gbm_probs['expected']:,.2f}",
                f"${gbm_probs['median']:,.2f}",
                f"{gbm_probs['P(gain)']:.1f}%",
                f"{gbm_risk['VaR_95']:.2f}%",
                f"{gbm_risk['CVaR_95']:.2f}%",
                f"{gbm_probs['P(+5%)']:.1f}%",
                f"{gbm_probs['P(-5%)']:.1f}%",
            ],
            'Regime-Switching': [
                f"${rs_probs['expected']:,.2f}",
                f"${rs_probs['median']:,.2f}",
                f"{rs_probs['P(gain)']:.1f}%",
                f"{rs_risk['VaR_95']:.2f}%",
                f"{rs_risk['CVaR_95']:.2f}%",
                f"{rs_probs['P(+5%)']:.1f}%",
                f"{rs_probs['P(-5%)']:.1f}%",
            ],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True, height=290)

        # ══════════════════════════════════════════
        # SECTION 5 — SENSITIVITY GRID
        # ══════════════════════════════════════════
        section("🎛  PARAMETER SENSITIVITY GRID (3×3)")
        st.caption("Varying σ (volatility) and μ (drift) around historical estimates")

        with st.spinner("Computing 9 sensitivity scenarios…"):
            grid = GBMSimulator.sensitivity_grid(
                start_price, mu_daily, sigma_daily,
                n_steps=mc_horizon, n_paths=min(mc_paths, 2000))

        # 3x3 grid of mini fan charts
        for row_i in range(3):
            cols_g = st.columns(3)
            for col_i in range(3):
                gi = row_i * 3 + col_i
                g = grid[gi]
                with cols_g[col_i]:
                    is_base = (row_i == 1 and col_i == 1)
                    border_col = '#00D4FF' if is_base else 'rgba(59,130,246,0.15)'
                    pg = g['probs']['P(gain)']
                    pg_col = '#22C993' if pg > 55 else ('#E84040' if pg < 45 else '#E8A838')
                    st.markdown(f"""
<div style="background:#111C2D;border:1px solid {border_col};
            border-radius:8px;padding:12px 14px;margin-bottom:6px">
  <div style="font-size:10px;font-weight:700;color:{'#00D4FF' if is_base else '#7A9BB5'};
              text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
    {'★ ' if is_base else ''}{g['label']}
  </div>
  <div style="font-family:'JetBrains Mono';font-size:13px;color:#E2EAF4">
    E[P] = ${g['probs']['expected']:,.0f}
    <span style="color:{pg_col};margin-left:8px">P(gain)={pg:.0f}%</span>
  </div>
</div>""", unsafe_allow_html=True)

                    s = g['stats']
                    days_s = list(range(len(s['median'])))
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(
                        x=days_s + days_s[::-1],
                        y=list(s['p95']) + list(s['p5'][::-1]),
                        fill='toself', fillcolor='rgba(59,130,246,0.10)',
                        line=dict(width=0), showlegend=False))
                    fig_s.add_trace(go.Scatter(
                        x=days_s, y=s['median'], mode='lines',
                        line=dict(color='#00D4FF', width=2), showlegend=False))
                    fig_s.add_hline(y=start_price,
                        line=dict(color='rgba(255,255,255,0.2)', dash='dot', width=1))
                    fig_s.update_layout(
                        height=180, template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(17,28,45,1)',
                        font=dict(color='#7A9BB5', size=9),
                        margin=dict(l=40, r=10, t=8, b=30),
                        xaxis=dict(gridcolor='rgba(59,130,246,0.06)', zeroline=False,
                                   showticklabels=True, tickfont=dict(size=8)),
                        yaxis=dict(gridcolor='rgba(59,130,246,0.06)', zeroline=False,
                                   tickfont=dict(size=8)),
                    )
                    st.plotly_chart(fig_s, use_container_width=True)

        # Sensitivity summary table
        section("📋  SENSITIVITY SUMMARY TABLE")
        sens_rows = []
        for g in grid:
            sens_rows.append({
                'Scenario': g['label'],
                'σ daily': f"{g['sigma']*100:.3f}%",
                'μ daily': f"{g['mu']*100:.4f}%",
                'E[Price]': f"${g['probs']['expected']:,.2f}",
                'Median': f"${g['probs']['median']:,.2f}",
                'P(Gain)': f"{g['probs']['P(gain)']:.1f}%",
                'P(+5%)': f"{g['probs']['P(+5%)']:.1f}%",
                'P(−5%)': f"{g['probs']['P(-5%)']:.1f}%",
            })
        st.dataframe(pd.DataFrame(sens_rows), use_container_width=True,
                     hide_index=True, height=370)

        # Final distribution histogram
        section("📊  FINAL PRICE DISTRIBUTION")
        finals = gbm_paths[:, -1]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=finals, nbinsx=60,
            marker_color='#3B82F6', opacity=0.75, name='Final Prices'))
        fig_hist.add_vline(x=start_price,
            line=dict(color='rgba(255,255,255,0.6)', dash='dash', width=2),
            annotation_text=f'Current: ${start_price:,.0f}',
            annotation_font=dict(color='#E2EAF4', size=10))
        fig_hist.add_vline(x=gbm_probs['expected'],
            line=dict(color='#00D4FF', dash='solid', width=2),
            annotation_text=f'Expected: ${gbm_probs["expected"]:,.0f}',
            annotation_font=dict(color='#00D4FF', size=10))
        fig_hist.add_vline(x=np.percentile(finals, 5),
            line=dict(color='#E84040', dash='dot', width=1.5),
            annotation_text='P5', annotation_font=dict(color='#E84040', size=9))
        fig_hist.add_vline(x=np.percentile(finals, 95),
            line=dict(color='#22C993', dash='dot', width=1.5),
            annotation_text='P95', annotation_font=dict(color='#22C993', size=9))
        pct_chart(fig_hist, height=320,
                  xaxis=dict(title='Final Price ($)', gridcolor='rgba(59,130,246,0.08)'),
                  yaxis=dict(title='Count', gridcolor='rgba(59,130,246,0.08)'))

        # Formula reference expander
        with st.expander("ℹ️  Monte Carlo Formula Reference (click to expand)"):
            st.markdown("""
| Formula | Expression |
|---|---|
| **GBM Daily Step** | `P(t+1) = P(t) × exp((μ − 0.5σ²)·Δt + σ·√Δt·Z)` |
| **Log Return** | `r_t = ln(P_t / P_{t-1})` |
| **Daily μ from annual** | `μ_daily = μ_annual / 252` |
| **Daily σ from annual** | `σ_daily = σ_annual / √252` |
| **VaR 95%** | `−Percentile(PnL, 5%)` |
| **CVaR 95%** | `−Mean(PnL where PnL ≤ VaR)` |
| **Regime-Switching** | Draw regime from transition matrix A, then draw returns from N(μ_k, Σ_k) |
""")
    else:
        st.info("▶  Run the pipeline first (🚀 Run RegimeEngine) to enable Monte Carlo simulations.")
