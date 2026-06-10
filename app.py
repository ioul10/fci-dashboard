# ── app.py ───────────────────────────────────────────
# Modèle de suivi d'un marché à terme
# sur indice boursier
# Futures Confidence Index (FCI) — S&P 500

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from src.data import load_data, get_summary_stats
from src.fci import compute_fci_rolling
from components.overview import render_overview
from components.composantes import render_composantes
from components.regimes import render_regimes
from components.data_view import render_data_view

# ── Configuration ────────────────────────────────────
st.set_page_config(
    page_title="FCI — Marché à Terme",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8b949e;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        color: #ffffff;
    }
    .stSlider > div > div > div {
        background-color: #58a6ff;
    }
    div[data-testid="stMetricValue"] {
        color: #58a6ff;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:

    # Logo + titre
    st.markdown("""
    <div style='
        text-align: center;
        padding: 20px 0 16px 0;
        border-bottom: 1px solid #30363d;
        margin-bottom: 20px;'>
        <div style='font-size: 2.5rem;'>
            📈
        </div>
        <div style='
            font-size: 1rem;
            font-weight: 700;
            color: #ffffff;
            margin-top: 8px;
            line-height: 1.3;'>
            Modèle de suivi<br>
            Marché à Terme
        </div>
        <div style='
            font-size: 0.75rem;
            color: #8b949e;
            margin-top: 6px;'>
            S&P 500 · FCI Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Période
    st.markdown("""
    <div style='
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;'>
        Période d'analyse
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox(
            'Début',
            [2019, 2020, 2021],
            index=0,
            label_visibility='collapsed')
    with col2:
        end_year = st.selectbox(
            'Fin',
            [2024, 2023, 2022],
            index=0,
            label_visibility='collapsed')

    start_date = f"{start_year}-01-01"
    end_date   = f"{end_year}-12-31"

    st.markdown("""
    <div style='
        font-size: 0.7rem;
        color: #6e7681;
        margin: 4px 0 16px 0;'>
        ← Début &nbsp;&nbsp;&nbsp; Fin →
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='
        border-top: 1px solid #30363d;
        margin: 8px 0 16px 0;'>
    </div>
    """, unsafe_allow_html=True)

    # Fenêtre rolling
    st.markdown("""
    <div style='
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;'>
        Fenêtre glissante
    </div>
    """, unsafe_allow_html=True)

    window = st.slider(
        'Fenêtre',
        min_value=120,
        max_value=504,
        value=252,
        step=21,
        label_visibility='collapsed',
        help="252 jours = 1 an de trading")

    st.markdown(f"""
    <div style='
        font-size: 0.75rem;
        color: #58a6ff;
        text-align: center;
        margin: 4px 0 16px 0;'>
        {window} jours ouvrés
        ({window//21} mois)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='
        border-top: 1px solid #30363d;
        margin: 8px 0 16px 0;'>
    </div>
    """, unsafe_allow_html=True)

    # Poids composantes
    st.markdown("""
    <div style='
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;'>
        Poids des composantes
    </div>
    """, unsafe_allow_html=True)

    w_is      = st.slider(
        '📡 IS',      0, 100, 30)
    w_vecm    = st.slider(
        '⚖️ VECM',    0, 100, 25)
    w_granger = st.slider(
        '🔗 Granger', 0, 100, 15)
    w_vix     = st.slider(
        '🌊 VIX',     0, 100, 30)

    total_w = w_is + w_vecm + w_granger + w_vix

    if total_w == 0:
        total_w = 1

    # Normalisation
    w_is_n      = w_is      / total_w
    w_vecm_n    = w_vecm    / total_w
    w_granger_n = w_granger / total_w
    w_vix_n     = w_vix     / total_w

    # Indicateur somme
    color_sum = ('#00C851'
                 if total_w == 100
                 else '#ff8800')
    st.markdown(f"""
    <div style='
        text-align: center;
        font-size: 0.8rem;
        color: {color_sum};
        margin-top: 8px;
        padding: 6px;
        background-color: #161b22;
        border-radius: 6px;
        border: 1px solid {color_sum};'>
        Σ = {total_w}%
        {'✅' if total_w == 100 else '⚠️'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='
        border-top: 1px solid #30363d;
        margin: 16px 0 12px 0;'>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style='
        font-size: 0.7rem;
        color: #6e7681;
        text-align: center;
        line-height: 1.6;'>
        Données · Yahoo Finance<br>
        Mise à jour · Quotidienne<br>
        <br>
        <span style='color: #30363d;'>
        ─────────────────
        </span><br>
        <span style='color: #58a6ff;'>
        FCI — PFE CDG Capital
        </span>
    </div>
    """, unsafe_allow_html=True)


# ── CHARGEMENT DONNÉES ───────────────────────────────
@st.cache_data(ttl=86400,
               show_spinner=False)
def get_data_cached(
        start: str,
        end: str):
    return load_data(start, end)


@st.cache_data(ttl=86400,
               show_spinner=False)
def get_fci_cached(
        start: str,
        end: str,
        window: int,
        w_is: float,
        w_vecm: float,
        w_granger: float,
        w_vix: float):
    data    = get_data_cached(start, end)
    weights = {
        'IS'      : w_is,
        'VECM'    : w_vecm,
        'Granger' : w_granger,
        'VIX'     : w_vix,
    }
    return compute_fci_rolling(
        data,
        window=window,
        weights=weights)


# ── LOADING ──────────────────────────────────────────
placeholder = st.empty()

with placeholder.container():
    st.markdown("""
    <div style='
        text-align: center;
        padding: 80px 40px;'>
        <div style='font-size: 3rem;
                    margin-bottom: 16px;'>
            📈
        </div>
        <div style='
            font-size: 1.2rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 8px;'>
            Modèle de suivi —
            Marché à Terme
        </div>
        <div style='
            font-size: 0.9rem;
            color: #8b949e;
            margin-bottom: 24px;'>
            Chargement des données
            et calcul du FCI...
        </div>
    </div>
    """, unsafe_allow_html=True)

    prog = st.progress(0)

    prog.progress(20)
    data  = get_data_cached(
        start_date, end_date)

    prog.progress(50)
    stats = get_summary_stats(data)

    prog.progress(70)
    df_fci = get_fci_cached(
        start_date, end_date,
        window,
        w_is_n, w_vecm_n,
        w_granger_n, w_vix_n)

    prog.progress(100)

placeholder.empty()


# ── HEADER ───────────────────────────────────────────
st.markdown(f"""
<div style='
    padding: 8px 0 20px 0;
    border-bottom: 1px solid #30363d;
    margin-bottom: 20px;'>
    <div style='
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;'>
        Modèle de suivi d'un marché
        à terme sur indice boursier
    </div>
    <div style='
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 4px;'>
        S&P 500 · E-mini Futures ·
        {stats['start']} →
        {stats['end']} ·
        {stats['n_obs']:,} observations ·
        Fenêtre {window}j
    </div>
</div>
""", unsafe_allow_html=True)


# ── TABS ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Vue générale",
    "🔬  Composantes",
    "🌊  Régimes",
    "📋  Données",
])

with tab1:
    render_overview(data, df_fci, stats)

with tab2:
    render_composantes(df_fci)

with tab3:
    render_regimes(data, df_fci['FCI'])

with tab4:
    render_data_view(
        data, df_fci, stats,
        start_date, end_date)
