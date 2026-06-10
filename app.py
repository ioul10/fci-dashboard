# ── app.py ───────────────────────────────────────────
# Modèle de suivi d'un marché à terme sur indice boursier
# Futures Confidence Index (FCI) — S&P 500

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from src.data import load_data, get_summary_stats
from src.fci import (compute_fci_rolling,
                     interpret_fci)
from src.regimes import (compute_vix_regimes,
                          compute_markov_regimes,
                          get_regime_stats)

# ── Configuration de la page ─────────────────────────
st.set_page_config(
    page_title="FCI — Marché à Terme",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS Dark Theme ───────────────────────────────────
st.markdown("""
<style>
    /* Background principal */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Cards métriques */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -1px;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .metric-sub {
        font-size: 0.9rem;
        color: #8b949e;
        margin-top: 8px;
    }

    /* Score card principal */
    .score-card {
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin: 16px 0;
    }

    .score-value {
        font-size: 5rem;
        font-weight: 800;
        letter-spacing: -2px;
    }

    .score-label {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 8px;
    }

    /* Composante cards */
    .comp-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    .comp-value {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .comp-name {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .comp-weight {
        font-size: 0.75rem;
        color: #6e7681;
        margin-top: 2px;
    }

    /* Titre principal */
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .main-subtitle {
        font-size: 0.95rem;
        color: #8b949e;
        margin-bottom: 24px;
    }

    /* Section titles */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #30363d;
    }

    /* Tabs */
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
        padding: 8px 16px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        color: #ffffff;
    }

    /* Slider */
    .stSlider {
        color: #58a6ff;
    }

    /* Divider */
    hr {
        border-color: #30363d;
        margin: 16px 0;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Fonctions utilitaires ────────────────────────────

@st.cache_data(ttl=3600)
def get_data(start: str, end: str) -> pd.DataFrame:
    return load_data(start, end)


@st.cache_data(ttl=3600)
def get_fci(start: str, end: str,
            window: int,
            w_is: float, w_vecm: float,
            w_granger: float,
            w_vix: float) -> pd.DataFrame:
    data = get_data(start, end)
    weights = {
        'IS'      : w_is,
        'VECM'    : w_vecm,
        'Granger' : w_granger,
        'VIX'     : w_vix,
    }
    return compute_fci_rolling(
        data, window=window, weights=weights)


def make_fci_chart(fci_series: pd.Series,
                   vix_series: pd.Series) -> go.Figure:
    """Graphique FCI rolling principal"""

    interp = fci_series.apply(
        lambda x: interpret_fci(x)['color'])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04)

    # Zone de confiance
    fig.add_hrect(y0=0.6, y1=1.0,
                  fillcolor='#00C851',
                  opacity=0.05,
                  line_width=0, row=1, col=1)
    fig.add_hrect(y0=0.4, y1=0.6,
                  fillcolor='#ff8800',
                  opacity=0.05,
                  line_width=0, row=1, col=1)
    fig.add_hrect(y0=0.0, y1=0.4,
                  fillcolor='#ff4444',
                  opacity=0.05,
                  line_width=0, row=1, col=1)

    # Ligne FCI
    fig.add_trace(
        go.Scatter(
            x=fci_series.index,
            y=fci_series.values,
            mode='lines',
            name='FCI',
            line=dict(color='#58a6ff',
                      width=2),
            fill='tozeroy',
            fillcolor='rgba(88, 166, 255, 0.05)',
            hovertemplate=(
                '<b>%{x|%d %b %Y}</b><br>'
                'FCI : %{y:.3f}<br>'
                '<extra></extra>')),
        row=1, col=1)

    # Lignes seuils
    for y, color, label in [
        (0.8, '#00C851', 'Très élevée'),
        (0.6, '#ffbb33', 'Élevée'),
        (0.4, '#ff8800', 'Modérée'),
        (0.2, '#ff4444', 'Faible'),
    ]:
        fig.add_hline(
            y=y, line_dash='dot',
            line_color=color,
            line_width=1,
            opacity=0.4,
            row=1, col=1)

    # Min / Max annotations
    idx_min = fci_series.idxmin()
    idx_max = fci_series.idxmax()

    fig.add_annotation(
        x=idx_min, y=fci_series.min(),
        text=f"Min {fci_series.min():.3f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor='#ff4444',
        font=dict(color='#ff4444', size=11),
        bgcolor='#1a0000',
        bordercolor='#ff4444',
        row=1, col=1)

    fig.add_annotation(
        x=idx_max, y=fci_series.max(),
        text=f"Max {fci_series.max():.3f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor='#00C851',
        font=dict(color='#00C851', size=11),
        bgcolor='#0d2b1a',
        bordercolor='#00C851',
        row=1, col=1)

    # VIX
    fig.add_trace(
        go.Scatter(
            x=vix_series.index,
            y=vix_series.values,
            mode='lines',
            name='VIX',
            line=dict(color='#f0883e',
                      width=1.5),
            hovertemplate=(
                '<b>%{x|%d %b %Y}</b><br>'
                'VIX : %{y:.1f}<br>'
                '<extra></extra>')),
        row=2, col=1)

    fig.add_hline(y=20, line_dash='dot',
                  line_color='#00C851',
                  line_width=1,
                  opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash='dot',
                  line_color='#ff4444',
                  line_width=1,
                  opacity=0.5, row=2, col=1)

    fig.update_layout(
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#c9d1d9',
                  family='Inter, sans-serif'),
        legend=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            borderwidth=1),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        hovermode='x unified',
        xaxis=dict(
            gridcolor='#21262d',
            showgrid=True),
        yaxis=dict(
            gridcolor='#21262d',
            showgrid=True,
            range=[0, 1],
            title='FCI'),
        xaxis2=dict(
            gridcolor='#21262d',
            showgrid=True),
        yaxis2=dict(
            gridcolor='#21262d',
            showgrid=True,
            title='VIX'),
    )

    return fig


def make_composantes_chart(
        df_fci: pd.DataFrame) -> go.Figure:
    """Graphique des 4 composantes"""

    colors = {
        'IS'      : '#58a6ff',
        'VECM'    : '#56d364',
        'Granger' : '#bc8cff',
        'VIX'     : '#f0883e',
    }

    fig = go.Figure()

    for comp, color in colors.items():
        fig.add_trace(go.Scatter(
            x=df_fci.index,
            y=df_fci[comp],
            mode='lines',
            name=comp,
            line=dict(color=color, width=1.5),
            hovertemplate=(
                f'<b>{comp}</b><br>'
                '%{x|%d %b %Y}<br>'
                'Score : %{y:.3f}<br>'
                '<extra></extra>')))

    fig.add_hline(y=0.5, line_dash='dot',
                  line_color='#8b949e',
                  line_width=1, opacity=0.5)

    fig.update_layout(
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#c9d1d9'),
        legend=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            borderwidth=1,
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        height=350,
        hovermode='x unified',
        xaxis=dict(gridcolor='#21262d'),
        yaxis=dict(gridcolor='#21262d',
                   range=[0, 1],
                   title='Score'),
    )

    return fig


def make_regime_chart(
        data: pd.DataFrame,
        fci_series: pd.Series) -> go.Figure:
    """Graphique régimes VIX + FCI"""

    vix_aligned = data['vix'].reindex(
        fci_series.index)

    def get_color(vix):
        if vix < 20:
            return 'rgba(0, 200, 81, 0.15)'
        elif vix < 30:
            return 'rgba(255, 187, 51, 0.15)'
        else:
            return 'rgba(255, 68, 68, 0.15)'

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.04)

    # FCI
    fig.add_trace(
        go.Scatter(
            x=fci_series.index,
            y=fci_series.values,
            mode='lines',
            name='FCI',
            line=dict(color='#58a6ff',
                      width=2)),
        row=1, col=1)

    # VIX
    fig.add_trace(
        go.Scatter(
            x=vix_aligned.index,
            y=vix_aligned.values,
            mode='lines',
            name='VIX',
            line=dict(color='#f0883e',
                      width=1.5),
            fill='tozeroy',
            fillcolor='rgba(240, 136, 62, 0.1)'),
        row=2, col=1)

    # Seuils VIX
    for y, color in [(20, '#00C851'),
                      (30, '#ff4444')]:
        fig.add_hline(
            y=y, line_dash='dot',
            line_color=color,
            line_width=1,
            opacity=0.6, row=2, col=1)

    fig.update_layout(
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#c9d1d9'),
        legend=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            borderwidth=1),
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
        hovermode='x unified',
        xaxis=dict(gridcolor='#21262d'),
        yaxis=dict(gridcolor='#21262d',
                   range=[0, 1],
                   title='FCI'),
        xaxis2=dict(gridcolor='#21262d'),
        yaxis2=dict(gridcolor='#21262d',
                    title='VIX'),
    )

    return fig


# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div style='text-align:center; padding: 16px 0;'>
        <div style='font-size:2rem;'>📈</div>
        <div style='font-size:1rem; font-weight:700;
                    color:#ffffff; margin-top:8px;'>
            FCI Dashboard
        </div>
        <div style='font-size:0.75rem; color:#8b949e;
                    margin-top:4px;'>
            Marché à Terme — S&P 500
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Période
    st.markdown(
        "<div class='section-title'>Période</div>",
        unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox(
            'Début', [2019, 2020, 2021],
            index=0)
    with col2:
        end_year = st.selectbox(
            'Fin', [2024, 2023, 2022],
            index=0)

    start_date = f"{start_year}-01-01"
    end_date   = f"{end_year}-12-31"

    st.markdown("---")

    # Fenêtre rolling
    st.markdown(
        "<div class='section-title'>"
        "Fenêtre glissante</div>",
        unsafe_allow_html=True)

    window = st.slider(
        'Jours ouvrés', 120, 504,
        252, step=21,
        help="252 = 1 an de trading (défaut)")

    st.markdown("---")

    # Poids des composantes
    st.markdown(
        "<div class='section-title'>"
        "Poids des composantes</div>",
        unsafe_allow_html=True)

    w_is      = st.slider('IS',      0, 100, 30)
    w_vecm    = st.slider('VECM',    0, 100, 25)
    w_granger = st.slider('Granger', 0, 100, 15)
    w_vix     = st.slider('VIX',     0, 100, 30)

    total_w = w_is + w_vecm + w_granger + w_vix

    if total_w != 100:
        st.warning(f"⚠️ Somme des poids = "
                   f"{total_w}% (doit être 100%)")
        w_is_n      = w_is / total_w
        w_vecm_n    = w_vecm / total_w
        w_granger_n = w_granger / total_w
        w_vix_n     = w_vix / total_w
    else:
        w_is_n      = w_is / 100
        w_vecm_n    = w_vecm / 100
        w_granger_n = w_granger / 100
        w_vix_n     = w_vix / 100

    st.markdown("---")

    st.markdown("""
    <div style='font-size:0.7rem; color:#6e7681;
                text-align:center;'>
        Données via Yahoo Finance<br>
        Mise à jour : quotidienne
    </div>
    """, unsafe_allow_html=True)


# ── CHARGEMENT DES DONNÉES ───────────────────────────
with st.spinner('Chargement des données...'):
    data = get_data(start_date, end_date)
    stats = get_summary_stats(data)

# ── CALCUL FCI ───────────────────────────────────────
with st.spinner('Calcul du FCI rolling... '
                '(2-3 minutes)'):
    df_fci = get_fci(
        start_date, end_date, window,
        w_is_n, w_vecm_n,
        w_granger_n, w_vix_n)

fci_series  = df_fci['FCI']
last_fci    = float(fci_series.iloc[-1])
last_interp = interpret_fci(last_fci)

# ── HEADER ───────────────────────────────────────────
st.markdown(f"""
<div class='main-title'>
    Modèle de suivi d'un marché à terme
    sur indice boursier
</div>
<div class='main-subtitle'>
    S&P 500 · Futures Confidence Index (FCI) ·
    {stats['start']} → {stats['end']} ·
    {stats['n_obs']:,} observations
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Vue générale",
    "🔬  Composantes",
    "🌊  Régimes",
    "📋  Données",
])

# ════════════════════════════════════════════════════
# TAB 1 — VUE GÉNÉRALE
# ════════════════════════════════════════════════════
with tab1:

    # Score principal + métriques
    col_score, col_metrics = st.columns(
        [1, 2], gap='large')

    with col_score:
        st.markdown(f"""
        <div class='score-card'
             style='background-color:
             {last_interp["bg"]};
             border: 2px solid
             {last_interp["color"]};'>
            <div style='font-size:0.8rem;
                        color:#8b949e;
                        text-transform:uppercase;
                        letter-spacing:1.5px;'>
                FCI du jour
            </div>
            <div class='score-value'
                 style='color:
                 {last_interp["color"]};'>
                {last_fci:.3f}
            </div>
            <div class='score-label'
                 style='color:
                 {last_interp["color"]};'>
                {last_interp["emoji"]}
                Confiance {last_interp["label"]}
            </div>
            <div style='font-size:0.8rem;
                        color:#8b949e;
                        margin-top:12px;'>
                {fci_series.index[-1].strftime(
                    "%d %B %Y")}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:

        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)

        with m1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'
                     style='color:#ff4444;
                     font-size:2rem;'>
                    {fci_series.min():.3f}
                </div>
                <div class='metric-label'>
                    FCI Minimum
                </div>
                <div class='metric-sub'>
                    {fci_series.idxmin().strftime(
                        "%d %b %Y")}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'
                     style='color:#00C851;
                     font-size:2rem;'>
                    {fci_series.max():.3f}
                </div>
                <div class='metric-label'>
                    FCI Maximum
                </div>
                <div class='metric-sub'>
                    {fci_series.idxmax().strftime(
                        "%d %b %Y")}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'
                     style='color:#58a6ff;
                     font-size:2rem;'>
                    {fci_series.mean():.3f}
                </div>
                <div class='metric-label'>
                    FCI Moyen
                </div>
                <div class='metric-sub'>
                    Sur toute la période
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'
                     style='color:#f0883e;
                     font-size:2rem;'>
                    {stats['vix_max']:.0f}
                </div>
                <div class='metric-label'>
                    VIX Maximum
                </div>
                <div class='metric-sub'>
                    {stats['vix_max_date']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Graphique FCI principal
    st.markdown(
        "<div class='section-title'>"
        "Évolution du FCI</div>",
        unsafe_allow_html=True)

    vix_aligned = data['vix'].reindex(
        fci_series.index)
    st.plotly_chart(
        make_fci_chart(fci_series, vix_aligned),
        use_container_width=True)


# ════════════════════════════════════════════════════
# TAB 2 — COMPOSANTES
# ════════════════════════════════════════════════════
with tab2:

    # Cards composantes
    st.markdown(
        "<div class='section-title'>"
        "Scores actuels</div>",
        unsafe_allow_html=True)

    last = df_fci.iloc[-1]
    comp_colors = {
        'IS'      : '#58a6ff',
        'VECM'    : '#56d364',
        'Granger' : '#bc8cff',
        'VIX'     : '#f0883e',
    }
    comp_weights = {
        'IS'      : '30%',
        'VECM'    : '25%',
        'Granger' : '15%',
        'VIX'     : '30%',
    }
    comp_desc = {
        'IS'      : 'Information Share',
        'VECM'    : 'Error Correction',
        'Granger' : 'Causalité',
        'VIX'     : 'Volatilité',
    }

    cols = st.columns(4)
    for col, (comp, color) in zip(
            cols, comp_colors.items()):
        with col:
            val = float(last[comp])
            st.markdown(f"""
            <div class='comp-card'
                 style='border-top: 3px solid
                 {color};'>
                <div class='comp-name'>
                    {comp_desc[comp]}
                </div>
                <div class='comp-value'
                     style='color:{color};'>
                    {val:.3f}
                </div>
                <div class='comp-weight'>
                    Poids : {comp_weights[comp]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Graphique composantes
    st.markdown(
        "<div class='section-title'>"
        "Évolution des composantes</div>",
        unsafe_allow_html=True)

    st.plotly_chart(
        make_composantes_chart(df_fci),
        use_container_width=True)

    # Interprétation
    st.markdown(
        "<div class='section-title'>"
        "Interprétation</div>",
        unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"""
        **Information Share (IS · 30%)**
        Mesure la contribution des futures au
        prix efficient commun. Un score de
        **{float(last['IS']):.3f}** indique une
        contribution quasi-équilibrée entre
        spot et futures sur données journalières.

        **VECM (25%)**
        Mesure la vitesse d'ajustement vers
        l'équilibre long terme. Un score de
        **{float(last['VECM']):.3f}** confirme
        que la direction ECT est correcte —
        le spot corrige vers les futures.
        """)

    with col_b:
        st.markdown(f"""
        **Granger (15%)**
        Mesure la causalité directionnelle.
        Un score de **{float(last['Granger']):.3f}**
        reflète la bidirectionnalité inhérente
        aux données journalières.

        **VIX (30%)**
        Mesure les conditions de marché via
        la volatilité implicite inversée.
        Un score de **{float(last['VIX']):.3f}**
        indique des conditions de marché
        actuellement favorables.
        """)


# ════════════════════════════════════════════════════
# TAB 3 — RÉGIMES
# ════════════════════════════════════════════════════
with tab3:

    # Stats par régime
    st.markdown(
        "<div class='section-title'>"
        "FCI par régime de marché</div>",
        unsafe_allow_html=True)

    regime_stats = get_regime_stats(
        data, fci_series)

    cols_r = st.columns(3)
    regime_emojis = {
        'Stable' : '🟢',
        'Stress' : '🟡',
        'Crise'  : '🔴'
    }

    for col, (_, row) in zip(
            cols_r,
            regime_stats.iterrows()):
        with col:
            emoji = regime_emojis.get(
                row['Régime'], '')
            st.markdown(f"""
            <div class='metric-card'
                 style='border-top: 3px solid
                 {row["color"]};'>
                <div style='font-size:0.85rem;
                            color:#8b949e;
                            text-transform:uppercase;
                            letter-spacing:1px;'>
                    {emoji} {row['Régime']}
                </div>
                <div style='font-size:2rem;
                            font-weight:700;
                            color:{row["color"]};
                            margin:8px 0;'>
                    {row['FCI moyen']:.3f}
                </div>
                <div style='font-size:0.8rem;
                            color:#8b949e;'>
                    {row['Obs']} jours ·
                    VIX moy. {row['VIX moyen']:.1f}
                </div>
                <div style='font-size:0.75rem;
                            color:#6e7681;
                            margin-top:4px;'>
                    [{row['FCI min']:.3f} —
                    {row['FCI max']:.3f}]
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Graphique régimes
    st.markdown(
        "<div class='section-title'>"
        "FCI et VIX — Détection des régimes"
        "</div>",
        unsafe_allow_html=True)

    st.plotly_chart(
        make_regime_chart(data, fci_series),
        use_container_width=True)

    # Légende régimes
    st.markdown("""
    <div style='display:flex; gap:24px;
                margin-top:8px;'>
        <div style='font-size:0.85rem;
                    color:#8b949e;'>
            🟢 <b style='color:#00C851;'>Stable</b>
            &nbsp; VIX < 20
        </div>
        <div style='font-size:0.85rem;
                    color:#8b949e;'>
            🟡 <b style='color:#ffbb33;'>Stress</b>
            &nbsp; 20 ≤ VIX < 30
        </div>
        <div style='font-size:0.85rem;
                    color:#8b949e;'>
            🔴 <b style='color:#ff4444;'>Crise</b>
            &nbsp; VIX ≥ 30
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TAB 4 — DONNÉES
# ════════════════════════════════════════════════════
with tab4:

    st.markdown(
        "<div class='section-title'>"
        "Données brutes — FCI Rolling"
        "</div>",
        unsafe_allow_html=True)

    # Infos marché
    col_i1, col_i2, col_i3, col_i4 = \
        st.columns(4)

    with col_i1:
        st.metric("Observations",
                  f"{stats['n_obs']:,}")
    with col_i2:
        st.metric("Corrélation Spot-Futures",
                  f"{stats['corr']:.4f}")
    with col_i3:
        st.metric("VIX moyen",
                  f"{stats['vix_mean']:.2f}")
    with col_i4:
        st.metric("Fenêtre rolling",
                  f"{window} jours")

    # Tableau FCI
    df_display = df_fci[
        ['FCI', 'IS', 'VECM',
         'Granger', 'VIX']].copy()
    df_display.index = df_display.index.strftime(
        '%Y-%m-%d')
    df_display = df_display.round(4)
    df_display = df_display.sort_index(
        ascending=False)

    st.dataframe(
        df_display,
        use_container_width=True,
        height=400)

    # Téléchargement
    csv = df_display.to_csv()
    st.download_button(
        label="⬇️ Télécharger CSV",
        data=csv,
        file_name=f"FCI_SP500_"
                  f"{start_date}_{end_date}.csv",
        mime='text/csv')
