
"""
EVT VaR & Expected Shortfall — Streamlit App
The Mountain Path - World of Finance
Prof. V. Ravichandran
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EVT Risk Model | The Mountain Path",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans Pro', sans-serif;
    color: #e6f1ff;
}

.stApp {
    background: linear-gradient(135deg, #1a2332, #243447, #2a3f5f);
    min-height: 100vh;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #003366 100%);
    border-right: 2px solid #FFD700;
}
section[data-testid="stSidebar"] * {
    color: #e6f1ff !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectSlider label {
    color: #FFD700 !important;
    font-weight: 600;
}

/* Selectbox dropdown box */
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #1e3a5f !important;
    border: 1.5px solid #FFD700 !important;
    border-radius: 6px !important;
    color: #e6f1ff !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div:hover {
    border-color: #FFD700 !important;
    background-color: #264d7a !important;
}
/* Selected text inside selectbox */
section[data-testid="stSidebar"] .stSelectbox > div > div > div {
    color: #ffffff !important;
    font-weight: 500 !important;
}
/* Dropdown arrow icon */
section[data-testid="stSidebar"] .stSelectbox svg {
    fill: #FFD700 !important;
}
/* Dropdown option list */
section[data-testid="stSidebar"] [data-baseweb="select"] ul,
[data-baseweb="popover"] ul {
    background-color: #112240 !important;
    border: 1px solid #004d80 !important;
}
[data-baseweb="popover"] li {
    background-color: #112240 !important;
    color: #e6f1ff !important;
}
[data-baseweb="popover"] li:hover {
    background-color: #1e3a5f !important;
    color: #FFD700 !important;
}
/* Select slider (confidence level) */
section[data-testid="stSidebar"] .stSelectSlider > div > div {
    background-color: #1e3a5f !important;
    border: 1.5px solid #FFD700 !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] .stSelectSlider [data-baseweb="slider"] div {
    background-color: #FFD700 !important;
}
/* Number/range slider track */
section[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSlider"] div,
section[data-testid="stSidebar"] [role="slider"] {
    background-color: #FFD700 !important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background-color: #FFD700 !important;
}
/* Run button */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #003366, #004d80) !important;
    color: #FFD700 !important;
    border: 1.5px solid #FFD700 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 10px !important;
    transition: all 0.2s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #004d80, #0066cc) !important;
    color: #ffffff !important;
    border-color: #ffffff !important;
}

/* Hero */
.hero-container {
    background: linear-gradient(135deg, #003366, #004d80);
    border: 1px solid #FFD700;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    text-align: center;
}
.hero-container h1 {
    font-family: 'Playfair Display', serif;
    color: #FFD700;
    font-size: 2.2rem;
    margin: 0 0 6px 0;
}
.hero-container p {
    color: #ADD8E6;
    font-size: 1.05rem;
    margin: 0;
}
.hero-brand {
    font-size: 0.85rem;
    color: #8892b0;
    margin-top: 8px;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    background: #112240;
    border: 1px solid #004d80;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #FFD700; }
.metric-card .label {
    font-size: 0.78rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-card .value {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #FFD700;
}
.metric-card .sub {
    font-size: 0.8rem;
    color: #ADD8E6;
    margin-top: 4px;
}

/* Info cards */
.info-card {
    background: #112240;
    border: 1px solid #004d80;
    border-left: 4px solid #FFD700;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.info-card h4 {
    font-family: 'Playfair Display', serif;
    color: #FFD700;
    margin: 0 0 8px 0;
    font-size: 1rem;
}
.info-card p, .info-card li {
    color: #e6f1ff;
    font-size: 0.9rem;
    line-height: 1.6;
    margin: 0;
}

/* Alert boxes */
.alert-danger {
    background: rgba(220, 53, 69, 0.15);
    border: 1px solid #dc3545;
    border-radius: 8px;
    padding: 14px 18px;
    color: #ff6b7a;
    margin-bottom: 16px;
}
.alert-success {
    background: rgba(40, 167, 69, 0.15);
    border: 1px solid #28a745;
    border-radius: 8px;
    padding: 14px 18px;
    color: #6bcb7f;
    margin-bottom: 16px;
}

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    color: #ADD8E6;
    font-size: 1.25rem;
    border-bottom: 1px solid #004d80;
    padding-bottom: 8px;
    margin-bottom: 18px;
}

/* Tail risk badge */
.badge-high   { background: #dc3545; color: #fff; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-medium { background: #fd7e14; color: #fff; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
.badge-low    { background: #28a745; color: #fff; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }

/* Footer */
.footer {
    background: linear-gradient(90deg, #0a1628, #003366);
    border-top: 1px solid #FFD700;
    border-radius: 8px;
    padding: 16px 24px;
    text-align: center;
    margin-top: 32px;
    color: #8892b0;
    font-size: 0.85rem;
}
.footer a { color: #FFD700; text-decoration: none; margin: 0 10px; }
.footer a:hover { text-decoration: underline; }

/* Table */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.styled-table th {
    background: #003366;
    color: #FFD700;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
}
.styled-table td {
    background: #112240;
    color: #e6f1ff;
    padding: 9px 14px;
    border-bottom: 1px solid #1e3a5f;
}
.styled-table tr:hover td { background: #1a3a5c; }
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
NIFTY50_LIST = [
    ("^NSEI",          "Nifty 50 Index"),
    ("ADANIPORTS.NS",  "Adani Ports and SEZ"),
    ("ASIANPAINT.NS",  "Asian Paints"),
    ("AXISBANK.NS",    "Axis Bank"),
    ("BAJAJ-AUTO.NS",  "Bajaj Auto"),
    ("BAJFINANCE.NS",  "Bajaj Finance"),
    ("BAJAJFINSV.NS",  "Bajaj Finserv"),
    ("BHARTIARTL.NS",  "Bharti Airtel"),
    ("BPCL.NS",        "Bharat Petroleum"),
    ("BRITANNIA.NS",   "Britannia Industries"),
    ("CIPLA.NS",       "Cipla"),
    ("COALINDIA.NS",   "Coal India"),
    ("DIVISLAB.NS",    "Divi's Laboratories"),
    ("DRREDDY.NS",     "Dr. Reddy's Laboratories"),
    ("EICHERMOT.NS",   "Eicher Motors"),
    ("GRASIM.NS",      "Grasim Industries"),
    ("HCLTECH.NS",     "HCL Technologies"),
    ("HDFCBANK.NS",    "HDFC Bank"),
    ("HDFCLIFE.NS",    "HDFC Life Insurance"),
    ("HEROMOTOCO.NS",  "Hero MotoCorp"),
    ("HINDALCO.NS",    "Hindalco Industries"),
    ("HINDUNILVR.NS",  "Hindustan Unilever"),
    ("ICICIBANK.NS",   "ICICI Bank"),
    ("INDUSINDBK.NS",  "IndusInd Bank"),
    ("INFY.NS",        "Infosys"),
    ("ITC.NS",         "ITC"),
    ("JSWSTEEL.NS",    "JSW Steel"),
    ("KOTAKBANK.NS",   "Kotak Mahindra Bank"),
    ("LT.NS",          "Larsen & Toubro"),
    ("M&M.NS",         "Mahindra & Mahindra"),
    ("MARUTI.NS",      "Maruti Suzuki"),
    ("NESTLEIND.NS",   "Nestle India"),
    ("NTPC.NS",        "NTPC"),
    ("ONGC.NS",        "Oil and Natural Gas"),
    ("POWERGRID.NS",   "Power Grid Corporation"),
    ("RELIANCE.NS",    "Reliance Industries"),
    ("SBILIFE.NS",     "SBI Life Insurance"),
    ("SBIN.NS",        "State Bank of India"),
    ("SUNPHARMA.NS",   "Sun Pharmaceutical"),
    ("TATAMOTORS.NS",  "Tata Motors"),
    ("TATASTEEL.NS",   "Tata Steel"),
    ("TCS.NS",         "Tata Consultancy Services"),
    ("TECHM.NS",       "Tech Mahindra"),
    ("TITAN.NS",       "Titan Company"),
    ("ULTRACEMCO.NS",  "UltraTech Cement"),
    ("WIPRO.NS",       "Wipro"),
    ("APOLLOHOSP.NS",  "Apollo Hospitals"),
    ("LTIM.NS",        "LTIMindtree"),
    ("ADANIENT.NS",    "Adani Enterprises"),
    ("SBICARD.NS",     "SBI Cards"),
    ("UPL.NS",         "UPL"),
]

TICKER_MAP  = {name: ticker for ticker, name in NIFTY50_LIST}
TICKER_NAMES = [name for _, name in NIFTY50_LIST]

# ── EVT Core ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_prices(ticker: str, years: float) -> pd.Series:
    import yfinance as yf
    end   = datetime.now()
    start = end - timedelta(days=int(years * 365))
    hist  = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError(f"No data for {ticker!r}")
    return hist["Close"]


def get_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def fit_gpd(exceedances: np.ndarray):
    xi, _, sigma = stats.genpareto.fit(exceedances, floc=0)
    return float(xi), float(sigma)


def pot_var_es(losses, threshold, alpha):
    exceedances = losses[losses > threshold] - threshold
    n, n_u = len(losses), len(exceedances)
    if n_u < 5:
        var = np.quantile(losses, alpha)
        tail = losses[losses >= var]
        es   = tail.mean() if len(tail) > 0 else var
        return var, es, np.nan, np.nan, n_u
    xi, sigma = fit_gpd(exceedances)
    zeta = n_u / n
    if abs(xi) < 1e-8:
        var = threshold + sigma * np.log(zeta / (1 - alpha))
    else:
        var = threshold + (sigma / xi) * (((zeta / (1 - alpha)) ** xi) - 1)
    es = (var + sigma - xi * threshold) / (1 - xi) if xi < 1 else np.nan
    return float(var), float(es), float(xi), float(sigma), n_u


def historical_var_es(losses, alpha):
    var  = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    es   = tail.mean() if len(tail) > 0 else var
    return float(var), float(es)


def parametric_var_es(losses, alpha):
    mu, sigma = losses.mean(), losses.std()
    z   = stats.norm.ppf(alpha)
    var = mu + sigma * z
    es  = mu + sigma * stats.norm.pdf(z) / (1 - alpha)
    return float(var), float(es)


# ── Plot helpers ─────────────────────────────────────────────────────────────
DARK_BG    = "#1a2332"
CARD_BG    = "#112240"
GOLD       = "#FFD700"
LIGHTBLUE  = "#ADD8E6"
STEEL      = "#4a90b8"
DARK_RED   = "#c0392b"
CRIMSON    = "#e74c3c"

def make_figure(nrows=1, ncols=2, figsize=(13, 4.5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    for ax in (axes.flat if hasattr(axes, 'flat') else [axes]):
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=LIGHTBLUE, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a5f")
        ax.title.set_color(GOLD)
        ax.xaxis.label.set_color(LIGHTBLUE)
        ax.yaxis.label.set_color(LIGHTBLUE)
        ax.grid(True, alpha=0.2, color="#4a6080")
    plt.tight_layout()
    return fig, axes


def plot_loss_distribution(losses, threshold, var, es, ticker):
    fig, axes = make_figure(1, 2, (13, 4.5))
    ax1, ax2 = axes

    # Left: full loss distribution
    ax1.hist(losses, bins=60, density=True, color=STEEL, alpha=0.75,
             edgecolor="#243447", linewidth=0.4, label="Daily losses")
    ax1.axvline(threshold, color="#adb5bd", lw=1.8, ls="--",
                label=f"Threshold u = {threshold:.4f}")
    ax1.axvline(var, color=DARK_RED,  lw=2.2, label=f"VaR = {var:.4f}")
    ax1.axvline(es,  color=CRIMSON,   lw=2.0, ls=":", label=f"ES = {es:.4f}")
    ax1.fill_betweenx([0, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 0.1],
                      var, losses.max(), alpha=0.08, color=CRIMSON)
    ax1.set_xlabel("Loss (positive = loss)")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Loss Distribution — {ticker}")
    ax1.legend(fontsize=8, facecolor=CARD_BG, edgecolor="#4a6080",
               labelcolor=LIGHTBLUE)

    # Right: tail / GPD fit
    exceedances = losses[losses > threshold] - threshold
    ax2.hist(exceedances, bins=max(15, len(exceedances) // 5), density=True,
             color=STEEL, alpha=0.75, edgecolor="#243447", linewidth=0.4,
             label="Exceedances (L − u)")
    return fig, ax2, exceedances


def plot_gpd_tail(ax2, exceedances, xi, sigma, var, threshold):
    if len(exceedances) >= 10 and not np.isnan(xi):
        x_gpd  = np.linspace(0, exceedances.max() * 1.05, 300)
        gpd_pdf = stats.genpareto.pdf(x_gpd, xi, loc=0, scale=sigma)
        ax2.plot(x_gpd, gpd_pdf, color=GOLD, lw=2.2,
                 label=f"GPD fit (ξ={xi:.3f}, σ={sigma:.3f})")
    ax2.axvline(var - threshold, color=DARK_RED, lw=1.8,
                label=f"VaR − u = {var - threshold:.4f}")
    ax2.set_xlabel("Exceedance (loss − threshold)")
    ax2.set_ylabel("Density")
    ax2.set_title("Tail: Exceedances & GPD Fit")
    ax2.legend(fontsize=8, facecolor=CARD_BG, edgecolor="#4a6080",
               labelcolor=LIGHTBLUE)


def plot_return_series(prices, ticker):
    returns = get_log_returns(prices)
    fig, axes = make_figure(2, 1, (13, 6))
    ax1, ax2  = axes

    ax1.plot(prices.index, prices.values, color=GOLD, lw=1.2, alpha=0.9)
    ax1.set_title(f"Price Series — {ticker}")
    ax1.set_ylabel("Price (₹)")
    ax1.set_xlabel("")

    neg = returns < 0
    ax2.bar(returns.index[~neg], returns[~neg], color="#28a745", alpha=0.7, width=1)
    ax2.bar(returns.index[neg],  returns[neg],  color="#dc3545", alpha=0.7, width=1)
    ax2.set_title("Log Returns (green=gain, red=loss)")
    ax2.set_ylabel("Log Return")
    ax2.axhline(0, color="#adb5bd", lw=0.8)

    plt.tight_layout(h_pad=2.5)
    return fig


def plot_mean_excess(losses):
    """Mean excess plot to guide threshold selection."""
    qs = np.linspace(0.70, 0.97, 40)
    thresholds = np.quantile(losses, qs)
    mes = [losses[losses > t].mean() - t for t in thresholds]
    fig, ax = make_figure(1, 1, (6, 3.5))
    ax = ax if not isinstance(ax, np.ndarray) else ax.flat[0]
    ax.plot(thresholds, mes, color=GOLD, lw=2, marker="o", ms=4)
    ax.set_xlabel("Threshold u")
    ax.set_ylabel("Mean Excess e(u)")
    ax.set_title("Mean Excess Plot (threshold selection)")
    plt.tight_layout()
    return fig


def plot_quantile_comparison(losses, alpha_values, evt_vars, hist_vars, par_vars):
    fig, ax = make_figure(1, 1, (7, 4))
    ax = ax if not isinstance(ax, np.ndarray) else ax.flat[0]
    pcts = [a * 100 for a in alpha_values]
    ax.plot(pcts, [v * 100 for v in evt_vars],  color=GOLD,      lw=2, marker="o", ms=5, label="EVT (POT-GPD)")
    ax.plot(pcts, [v * 100 for v in hist_vars], color=LIGHTBLUE, lw=2, marker="s", ms=5, label="Historical")
    ax.plot(pcts, [v * 100 for v in par_vars],  color="#adb5bd", lw=2, marker="^", ms=5, label="Parametric (Normal)")
    ax.set_xlabel("Confidence Level (%)")
    ax.set_ylabel("VaR (%)")
    ax.set_title("VaR Comparison across Methods")
    ax.legend(fontsize=9, facecolor=CARD_BG, edgecolor="#4a6080", labelcolor=LIGHTBLUE)
    plt.tight_layout()
    return fig


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 12px 0 20px;'>
        <span style='font-family:Playfair Display,serif; color:#FFD700;
                     font-size:1.15rem; font-weight:700;'>
            🏔️ The Mountain Path
        </span><br>
        <span style='color:#8892b0; font-size:0.8rem;'>World of Finance</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:#FFD700;font-weight:600;font-size:0.9rem;">📌 SELECT STOCK</p>',
                unsafe_allow_html=True)
    selected_name = st.selectbox("Stock / Index", TICKER_NAMES, index=0, label_visibility="collapsed")
    selected_ticker = TICKER_MAP[selected_name]

    st.markdown("---")
    st.markdown('<p style="color:#FFD700;font-weight:600;font-size:0.9rem;">⚙️ MODEL PARAMETERS</p>',
                unsafe_allow_html=True)

    years = st.slider("Historical Data (years)", 1, 5, 3, key="years")

    alpha = st.select_slider(
        "Confidence Level (α)",
        options=[0.90, 0.95, 0.99, 0.999],
        value=0.95,
        format_func=lambda x: f"{x*100:.1f}%"
    )

    threshold_q = st.slider("Threshold Quantile (%)", 80, 97, 90, key="tq") / 100

    st.markdown("---")
    st.markdown('<p style="color:#FFD700;font-weight:600;font-size:0.9rem;">💡 ABOUT EVT</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem; color:#8892b0; line-height:1.6;'>
    <b style='color:#ADD8E6;'>Peaks-Over-Threshold (POT)</b> method fits a 
    <b style='color:#ADD8E6;'>Generalised Pareto Distribution (GPD)</b> to 
    exceedances above a high threshold — capturing tail risk beyond 
    normal distribution assumptions.<br><br>
    <b style='color:#ADD8E6;'>ξ (shape)</b> &gt; 0 → heavy tail<br>
    <b style='color:#ADD8E6;'>ξ = 0</b> → exponential tail<br>
    <b style='color:#ADD8E6;'>ξ &lt; 0</b> → bounded tail
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("▶ Run EVT Analysis", use_container_width=True,
                        type="primary")

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero-container'>
    <h1>📊 Extreme Value Theory — VaR & ES</h1>
    <p>Peaks-Over-Threshold (POT) · Generalised Pareto Distribution (GPD)</p>
    <p class='hero-brand'>
        Prof. V. Ravichandran &nbsp;|&nbsp;
        28+ Years Corporate Finance &amp; Banking Experience &nbsp;|&nbsp;
        10+ Years Academic Excellence
    </p>
</div>
""", unsafe_allow_html=True)

# ── Main logic ───────────────────────────────────────────────────────────────
if not run_btn and "evt_result" not in st.session_state:
    st.markdown("""
    <div class='info-card'>
        <h4>👈 Getting Started</h4>
        <p>Select a Nifty 50 stock or the index from the sidebar, configure your
        parameters, and click <b>Run EVT Analysis</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4>📐 What is EVT?</h4>
            <p>Extreme Value Theory focuses on the <b>statistical behaviour of
            extremes</b> — the worst losses that occur infrequently but with
            devastating impact. The POT-GPD approach is considered the gold 
            standard for tail risk measurement in financial risk management.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4>📏 Key Risk Measures</h4>
            <p><b>VaR (Value at Risk):</b> Maximum expected loss at a given 
            confidence level over one period.<br><br>
            <b>ES (Expected Shortfall):</b> Average loss in the worst scenarios 
            beyond VaR — satisfies sub-additivity, preferred under Basel III/IV.</p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ── Run analysis ─────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Fetching {selected_name} ({selected_ticker}) data…"):
        try:
            prices = fetch_prices(selected_ticker, years)
            returns = get_log_returns(prices)
            losses  = -np.asarray(returns, dtype=float)
            threshold = float(np.quantile(losses, threshold_q))
            var, es, xi, sigma, n_u = pot_var_es(losses, threshold, alpha)
            hist_var, hist_es = historical_var_es(losses, alpha)
            par_var,  par_es  = parametric_var_es(losses, alpha)

            st.session_state.evt_result = {
                "ticker": selected_ticker, "name": selected_name,
                "prices": prices, "returns": returns, "losses": losses,
                "threshold": threshold, "alpha": alpha,
                "var": var, "es": es, "xi": xi, "sigma": sigma, "n_u": n_u,
                "hist_var": hist_var, "hist_es": hist_es,
                "par_var": par_var, "par_es": par_es,
                "n_obs": len(losses),
            }
        except Exception as exc:
            st.markdown(f"<div class='alert-danger'>❌ {exc}</div>", unsafe_allow_html=True)
            st.stop()

r = st.session_state.get("evt_result")
if not r:
    st.stop()

# Derived
pct_label = f"{r['alpha']*100:.0f}%"
tail_type  = "Heavy tail (ξ > 0) ⚠️" if r['xi'] > 0.1 else (
             "Near-exponential (ξ ≈ 0)" if abs(r['xi']) <= 0.1 else "Bounded tail (ξ < 0)")

# ── Status bar ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='alert-success'>
    ✅ Analysis complete &nbsp;|&nbsp; <b>{r['name']}</b> ({r['ticker']}) &nbsp;|&nbsp;
    {r['n_obs']} trading days &nbsp;|&nbsp; α = {pct_label} &nbsp;|&nbsp;
    Threshold exceedances: {r['n_u']}
</div>
""", unsafe_allow_html=True)

# ── Metric cards ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='metric-grid'>
    <div class='metric-card'>
        <div class='label'>EVT VaR ({pct_label})</div>
        <div class='value'>{r['var']*100:.2f}%</div>
        <div class='sub'>POT-GPD estimate</div>
    </div>
    <div class='metric-card'>
        <div class='label'>Expected Shortfall ({pct_label})</div>
        <div class='value' style='color:#dc3545;'>{r['es']*100:.2f}%</div>
        <div class='sub'>CVaR / tail loss avg</div>
    </div>
    <div class='metric-card'>
        <div class='label'>GPD Shape ξ</div>
        <div class='value' style='color:{"#dc3545" if r["xi"]>0.1 else "#28a745"};'>{r['xi']:.4f}</div>
        <div class='sub'>{tail_type}</div>
    </div>
    <div class='metric-card'>
        <div class='label'>GPD Scale σ</div>
        <div class='value'>{r['sigma']:.4f}</div>
        <div class='sub'>Threshold: {r['threshold']:.4f}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Loss Distribution",
    "📈 Price & Returns",
    "📐 Mean Excess Plot",
    "⚖️ Model Comparison",
    "📋 Summary Report",
])

with tab1:
    st.markdown('<p class="section-header">Loss Distribution & GPD Tail Fit</p>',
                unsafe_allow_html=True)
    fig1, ax2, exceedances = plot_loss_distribution(
        r['losses'], r['threshold'], r['var'], r['es'], r['ticker'])
    plot_gpd_tail(ax2, exceedances, r['xi'], r['sigma'], r['var'], r['threshold'])
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

with tab2:
    st.markdown('<p class="section-header">Price Series & Log Returns</p>',
                unsafe_allow_html=True)
    fig2 = plot_return_series(r['prices'], r['ticker'])
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

with tab3:
    st.markdown('<p class="section-header">Mean Excess Plot — Threshold Diagnostics</p>',
                unsafe_allow_html=True)
    col_me, col_info = st.columns([2, 1])
    with col_me:
        fig3 = plot_mean_excess(r['losses'])
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
    with col_info:
        st.markdown("""
        <div class='info-card'>
            <h4>Reading the Plot</h4>
            <p>The mean excess function <b>e(u)</b> = E[L − u | L > u].<br><br>
            • <b>Linear increase</b> → GPD with ξ > 0 (heavy tail)<br>
            • <b>Horizontal</b> → exponential tail (ξ ≈ 0)<br>
            • <b>Linear decrease</b> → bounded tail (ξ < 0)<br><br>
            Choose the threshold <b>u</b> at the point where the plot 
            becomes approximately linear.</p>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown('<p class="section-header">VaR Comparison: EVT vs Historical vs Parametric</p>',
                unsafe_allow_html=True)
    alpha_values = [0.90, 0.95, 0.99, 0.999]
    evt_vars, hist_vars, par_vars = [], [], []
    evt_ess, hist_ess, par_ess   = [], [], []
    losses = r['losses']
    for a in alpha_values:
        v, e, _, _, _ = pot_var_es(losses, r['threshold'], a)
        evt_vars.append(v); evt_ess.append(e)
        hv, he = historical_var_es(losses, a)
        hist_vars.append(hv); hist_ess.append(he)
        pv, pe = parametric_var_es(losses, a)
        par_vars.append(pv); par_ess.append(pe)

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        fig4 = plot_quantile_comparison(losses, alpha_values, evt_vars, hist_vars, par_vars)
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

    with col_table:
        rows = ""
        for a, ev, hv, pv in zip(alpha_values, evt_vars, hist_vars, par_vars):
            rows += f"""<tr>
                <td>{a*100:.1f}%</td>
                <td style='color:#FFD700;'>{ev*100:.2f}%</td>
                <td>{hv*100:.2f}%</td>
                <td>{pv*100:.2f}%</td>
            </tr>"""
        st.markdown(f"""
        <table class='styled-table'>
            <thead><tr>
                <th>α</th><th>EVT VaR</th><th>Hist VaR</th><th>Norm VaR</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

with tab5:
    st.markdown('<p class="section-header">Full Risk Report</p>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='info-card'>
            <h4>📌 EVT (POT-GPD) Results</h4>
            <p>
            <b>Asset:</b> {r['name']} ({r['ticker']})<br>
            <b>Data period:</b> {r['prices'].index[0].date()} – {r['prices'].index[-1].date()}<br>
            <b>Observations:</b> {r['n_obs']} trading days<br>
            <b>Confidence level:</b> {pct_label}<br>
            <b>Threshold (u):</b> {r['threshold']:.4f} ({threshold_q*100:.0f}th percentile)<br>
            <b>Exceedances:</b> {r['n_u']}<br>
            <br>
            <b>VaR ({pct_label}):</b> {r['var']*100:.2f}%<br>
            <b>Expected Shortfall:</b> {r['es']*100:.2f}%<br>
            <b>GPD Shape (ξ):</b> {r['xi']:.4f}<br>
            <b>GPD Scale (σ):</b> {r['sigma']:.4f}<br>
            <b>Tail Type:</b> {tail_type}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='info-card'>
            <h4>⚖️ Method Comparison @ {pct_label}</h4>
            <p>
            <b style='color:#FFD700;'>EVT VaR:</b> {r['var']*100:.2f}% &nbsp;|&nbsp;
            <b style='color:#FFD700;'>EVT ES:</b> {r['es']*100:.2f}%<br>
            <b>Historical VaR:</b> {r['hist_var']*100:.2f}% &nbsp;|&nbsp;
            <b>Historical ES:</b> {r['hist_es']*100:.2f}%<br>
            <b>Parametric VaR:</b> {r['par_var']*100:.2f}% &nbsp;|&nbsp;
            <b>Parametric ES:</b> {r['par_es']*100:.2f}%<br>
            <br>
            <b>EVT vs Historical premium:</b> {(r['var']-r['hist_var'])*100:+.2f}%<br>
            <b>EVT vs Parametric premium:</b> {(r['var']-r['par_var'])*100:+.2f}%<br>
            <br>
            <b>Interpretation:</b> {"EVT captures fatter tails than the normal distribution, providing a more conservative (and realistic) risk estimate."
            if r['var'] > r['par_var'] else
            "The normal distribution slightly overestimates tail risk for this asset at this threshold."}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
        <h4>📚 Methodology Notes</h4>
        <p>
        <b>POT-GPD Framework:</b> Losses exceeding a high threshold u follow a 
        Generalised Pareto Distribution (Pickands–Balkema–de Haan theorem). 
        GPD parameters are estimated via maximum likelihood.<br><br>
        <b>VaR Formula:</b> u + (σ/ξ)·[(ζ/(1−α))^ξ − 1] where ζ = n_u/n<br>
        <b>ES Formula:</b> (VaR + σ − ξu) / (1 − ξ) [valid for ξ &lt; 1]<br><br>
        <b>Basel III/IV Relevance:</b> Expected Shortfall (CVaR) replaces VaR 
        as the primary regulatory risk measure under FRTB (SA and IMA), 
        computed at the 97.5% confidence level over 10-day holding periods.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    <b style='color:#FFD700;'>Prof. V. Ravichandran</b> &nbsp;|&nbsp;
    28+ Years Corporate Finance &amp; Banking &nbsp;|&nbsp; 10+ Years Academic Excellence<br>
    🏔️ The Mountain Path — World of Finance<br>
    <a href='https://www.linkedin.com/in/trichyravis' target='_blank'>🔗 LinkedIn</a>
    <a href='https://github.com/trichyravis' target='_blank'>🐙 GitHub</a>
    <br><span style='font-size:0.78rem; color:#4a6080;'>
    For educational purposes. Not investment advice.</span>
</div>
""", unsafe_allow_html=True)
