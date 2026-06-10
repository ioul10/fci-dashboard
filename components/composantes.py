# ── components/composantes.py ────────────────────────
# Page 2 — Composantes du FCI

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Config composantes ───────────────────────────────

COMP_CONFIG = {
    'IS' : {
        'name'   : 'Information Share',
        'color'  : '#58a6ff',
        'weight' : '30%',
        'desc'   : (
            'Mesure la contribution des futures '
            'au prix efficient commun. '
            'Basé sur Hasbrouck (1995).'),
        'icon'   : '📡',
    },
    'VECM' : {
        'name'   : 'Error Correction',
        'color'  : '#56d364',
        'weight' : '25%',
        'desc'   : (
            'Mesure la vitesse d\'ajustement '
            'vers l\'équilibre long terme. '
            'α_spot < 0 → futures mènent.'),
        'icon'   : '⚖️',
    },
    'Granger' : {
        'name'   : 'Causalité Granger',
        'color'  : '#bc8cff',
        'weight' : '15%',
        'desc'   : (
            'Mesure la causalité '
            'directionnelle entre spot '
            'et futures via F-statistiques.'),
        'icon'   : '🔗',
    },
    'VIX' : {
        'name'   : 'Volatilité (VIX)',
        'color'  : '#f0883e',
        'weight' : '30%',
        'desc'   : (
            'VIX inversé et normalisé. '
            'Capte les conditions de marché. '
            'VIX élevé → confiance faible.'),
        'icon'   : '🌊',
    },
}


def render_comp_cards(
        df_fci: pd.DataFrame) -> None:
    """
    Affiche les 4 cards de composantes
    avec score actuel
    """
    last = df_fci.iloc[-1]
    cols = st.columns(4, gap='small')

    for col, (comp, config) in zip(
            cols, COMP_CONFIG.items()):

        val     = float(last[comp])
        color   = config['color']
        # Barre de progression
        pct     = int(val * 100)

        with col:
            st.markdown(f"""
            <div style='
                background-color: #161b22;
                border: 1px solid #30363d;
                border-top: 3px solid {color};
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                height: 180px;'>

                <div style='font-size:1.5rem;'>
                    {config["icon"]}
                </div>

                <div style='
                    font-size: 0.7rem;
                    color: #8b949e;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin: 6px 0 4px 0;'>
                    {config["name"]}
                </div>

                <div style='
                    font-size: 2rem;
                    font-weight: 700;
                    color: {color};
                    line-height: 1.1;'>
                    {val:.3f}
                </div>

                <div style='
                    background-color: #21262d;
                    border-radius: 4px;
                    height: 4px;
                    margin: 10px 0 6px 0;
                    overflow: hidden;'>
                    <div style='
                        background-color: {color};
                        width: {pct}%;
                        height: 100%;
                        border-radius: 4px;'>
                    </div>
                </div>

                <div style='
                    font-size: 0.7rem;
                    color: #6e7681;'>
                    Poids : {config["weight"]}
                </div>

            </div>
            """, unsafe_allow_html=True)


def render_comp_chart(
        df_fci: pd.DataFrame) -> None:
    """
    Graphique évolution des 4 composantes
    """
    fig = make_subplots(
        rows=2, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
        subplot_titles=[
            'Information Share (IS)',
            'VECM — Error Correction',
            'Granger — Causalité',
            'VIX — Volatilité',
        ])

    positions = {
        'IS'      : (1, 1),
        'VECM'    : (1, 2),
        'Granger' : (2, 1),
        'VIX'     : (2, 2),
    }

    for comp, (row, col) in positions.items():
        color = COMP_CONFIG[comp]['color']

        fig.add_trace(
            go.Scatter(
                x=df_fci.index,
                y=df_fci[comp],
                mode='lines',
                name=comp,
                line=dict(
                    color=color,
                    width=1.5),
                fill='tozeroy',
                fillcolor=f'rgba('
                          f'{int(color[1:3], 16)},'
                          f'{int(color[3:5], 16)},'
                          f'{int(color[5:7], 16)},'
                          f'0.08)',
                hovertemplate=(
                    f'<b>{comp}</b><br>'
                    '%{x|%d %b %Y}<br>'
                    'Score : <b>%{y:.3f}</b>'
                    '<extra></extra>')),
            row=row, col=col)

        # Ligne 0.5
        fig.add_hline(
            y=0.5,
            line_dash='dot',
            line_color='#30363d',
            line_width=1,
            row=row, col=col)

    # Layout
    fig.update_layout(
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(
            color='#c9d1d9',
            family='Inter, sans-serif',
            size=11),
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        hovermode='x unified',
    )

    # Axes
    for i in range(1, 5):
        fig.update_yaxes(
            range=[0, 1],
            gridcolor='#21262d',
            showgrid=True,
            zeroline=False,
            row=(i-1)//2 + 1,
            col=(i-1) % 2 + 1)
        fig.update_xaxes(
            gridcolor='#21262d',
            showgrid=True,
            zeroline=False,
            row=(i-1)//2 + 1,
            col=(i-1) % 2 + 1)

    # Titres des sous-graphiques en couleur
    for i, (comp, config) in enumerate(
            COMP_CONFIG.items()):
        fig.layout.annotations[i].font.color = \
            config['color']
        fig.layout.annotations[i].font.size = 12

    st.plotly_chart(
        fig, use_container_width=True)


def render_comp_table(
        df_fci: pd.DataFrame) -> None:
    """
    Tableau des scores récents par composante
    """
    # 10 dernières observations
    recent = df_fci[
        ['FCI', 'IS', 'VECM',
         'Granger', 'VIX']
    ].tail(10).copy()

    recent.index = recent.index.strftime(
        '%d %b %Y')
    recent = recent.sort_index(ascending=False)
    recent = recent.round(4)

    st.dataframe(
        recent,
        use_container_width=True,
        height=300)


def render_comp_interpretation(
        df_fci: pd.DataFrame) -> None:
    """
    Interprétation des scores actuels
    """
    last = df_fci.iloc[-1]

    cols = st.columns(2, gap='large')

    interpretations = {
        'IS' : {
            'score' : float(last['IS']),
            'high'  : (
                'Les futures contribuent '
                'majoritairement au prix '
                'efficient → price discovery '
                'forte'),
            'low'   : (
                'Contribution équilibrée '
                'spot-futures → typique des '
                'données journalières'),
        },
        'VECM' : {
            'score' : float(last['VECM']),
            'high'  : (
                'Direction ECT correcte — '
                'le spot corrige vers les '
                'futures → futures mènent'),
            'low'   : (
                'Direction ECT ambiguë → '
                'relation long terme '
                'à surveiller'),
        },
        'Granger' : {
            'score' : float(last['Granger']),
            'high'  : (
                'Futures Granger-causent '
                'le spot de manière '
                'dominante'),
            'low'   : (
                'Causalité bidirectionnelle '
                '→ normal sur données '
                'journalières'),
        },
        'VIX' : {
            'score' : float(last['VIX']),
            'high'  : (
                'VIX faible → conditions '
                'de marché favorables → '
                'confiance élevée'),
            'low'   : (
                'VIX élevé → stress de '
                'marché → confiance '
                'dégradée'),
        },
    }

    items = list(interpretations.items())

    for col, chunk in zip(
            cols,
            [items[:2], items[2:]]):
        with col:
            for comp, info in chunk:
                config = COMP_CONFIG[comp]
                score  = info['score']
                msg    = (info['high']
                          if score >= 0.5
                          else info['low'])
                icon   = ('✅' if score >= 0.5
                          else '⚠️')

                st.markdown(f"""
                <div style='
                    background-color: #161b22;
                    border: 1px solid #30363d;
                    border-left: 3px solid
                    {config["color"]};
                    border-radius: 8px;
                    padding: 12px 16px;
                    margin-bottom: 10px;'>
                    <div style='
                        display: flex;
                        justify-content:
                        space-between;
                        align-items: center;'>
                        <span style='
                            font-size: 0.85rem;
                            font-weight: 600;
                            color:
                            {config["color"]};'>
                            {config["icon"]}
                            {comp}
                        </span>
                        <span style='
                            font-size: 1rem;
                            font-weight: 700;
                            color:
                            {config["color"]};'>
                            {score:.3f}
                        </span>
                    </div>
                    <div style='
                        font-size: 0.8rem;
                        color: #8b949e;
                        margin-top: 6px;'>
                        {icon} {msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_composantes(
        df_fci: pd.DataFrame) -> None:
    """
    Rendu complet de la page Composantes

    Parameters
    ----------
    df_fci : pd.DataFrame — FCI rolling
    """

    # Cards scores actuels
    st.markdown("""
    <div style='
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #30363d;'>
        Scores actuels
    </div>
    """, unsafe_allow_html=True)

    render_comp_cards(df_fci)

    st.markdown(
        "<br>", unsafe_allow_html=True)

    # Graphique composantes
    st.markdown("""
    <div style='
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #30363d;'>
        Évolution des composantes
    </div>
    """, unsafe_allow_html=True)

    render_comp_chart(df_fci)

    # Interprétation
    st.markdown("""
    <div style='
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #30363d;'>
        Interprétation
    </div>
    """, unsafe_allow_html=True)

    render_comp_interpretation(df_fci)
