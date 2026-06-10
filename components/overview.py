# ── components/overview.py ───────────────────────────
# Page 1 — Vue générale du FCI

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.fci import interpret_fci


def render_score_card(
        fci_series: pd.Series) -> None:
    """
    Affiche le score FCI du jour
    avec métriques clés
    """
    last_fci    = float(fci_series.iloc[-1])
    last_date   = fci_series.index[-1]
    last_interp = interpret_fci(last_fci)

    # Variation vs hier
    if len(fci_series) > 1:
        prev_fci = float(fci_series.iloc[-2])
        delta    = last_fci - prev_fci
        delta_str = (f"+{delta:.3f}"
                     if delta >= 0
                     else f"{delta:.3f}")
        delta_color = ('#00C851'
                       if delta >= 0
                       else '#ff4444')
    else:
        delta_str   = "—"
        delta_color = '#8b949e'

    # Layout score + métriques
    col_score, col_metrics = st.columns(
        [1, 2], gap='large')

    # ── Score principal ──────────────────────
    with col_score:
        st.markdown(f"""
        <div style='
            background-color:{last_interp["bg"]};
            border: 2px solid {last_interp["color"]};
            border-radius: 16px;
            padding: 32px 24px;
            text-align: center;'>

            <div style='
                font-size: 0.75rem;
                color: #8b949e;
                text-transform: uppercase;
                letter-spacing: 2px;
                margin-bottom: 12px;'>
                Futures Confidence Index
            </div>

            <div style='
                font-size: 4.5rem;
                font-weight: 800;
                color: {last_interp["color"]};
                letter-spacing: -2px;
                line-height: 1;'>
                {last_fci:.3f}
            </div>

            <div style='
                font-size: 1.1rem;
                font-weight: 600;
                color: {last_interp["color"]};
                margin-top: 12px;'>
                {last_interp["emoji"]}
                &nbsp;Confiance {last_interp["label"]}
            </div>

            <div style='
                font-size: 0.85rem;
                color: #8b949e;
                margin-top: 16px;
                padding-top: 16px;
                border-top: 1px solid #30363d;'>
                {last_date.strftime("%d %B %Y")}
            </div>

            <div style='
                font-size: 0.85rem;
                color: {delta_color};
                margin-top: 6px;'>
                {delta_str} vs hier
            </div>

        </div>
        """, unsafe_allow_html=True)

    # ── Métriques ────────────────────────────
    with col_metrics:

        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)

        metrics = [
            (m1, fci_series.min(),
             'FCI Minimum',
             fci_series.idxmin().strftime(
                 "%d %b %Y"),
             '#ff4444'),
            (m2, fci_series.max(),
             'FCI Maximum',
             fci_series.idxmax().strftime(
                 "%d %b %Y"),
             '#00C851'),
            (m3, fci_series.mean(),
             'FCI Moyen',
             'Sur toute la période',
             '#58a6ff'),
            (m4, fci_series.std(),
             'FCI Volatilité',
             'Écart-type',
             '#bc8cff'),
        ]

        for col, val, label, sub, color in metrics:
            with col:
                st.markdown(f"""
                <div style='
                    background-color: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 12px;
                    padding: 16px;
                    text-align: center;
                    margin: 4px 0;'>
                    <div style='
                        font-size: 2rem;
                        font-weight: 700;
                        color: {color};'>
                        {val:.3f}
                    </div>
                    <div style='
                        font-size: 0.75rem;
                        color: #8b949e;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        margin-top: 4px;'>
                        {label}
                    </div>
                    <div style='
                        font-size: 0.75rem;
                        color: #6e7681;
                        margin-top: 4px;'>
                        {sub}
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_fci_chart(
        fci_series: pd.Series,
        vix_series: pd.Series) -> None:
    """
    Graphique FCI rolling + VIX
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.03)

    # ── Zones de confiance ───────────────────
    zones = [
        (0.8, 1.0, 'rgba(0,200,81,0.04)'),
        (0.6, 0.8, 'rgba(255,187,51,0.04)'),
        (0.4, 0.6, 'rgba(255,136,0,0.04)'),
        (0.0, 0.4, 'rgba(255,68,68,0.04)'),
    ]
    for y0, y1, color in zones:
        fig.add_hrect(
            y0=y0, y1=y1,
            fillcolor=color,
            line_width=0,
            row=1, col=1)

    # ── FCI line ─────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=fci_series.index,
            y=fci_series.values,
            mode='lines',
            name='FCI',
            line=dict(
                color='#58a6ff',
                width=2),
            fill='tozeroy',
            fillcolor='rgba(88,166,255,0.06)',
            hovertemplate=(
                '<b>%{x|%d %b %Y}</b><br>'
                'FCI : <b>%{y:.3f}</b>'
                '<extra></extra>')),
        row=1, col=1)

    # ── Seuils FCI ───────────────────────────
    seuils = [
        (0.8, '#00C851', 'Très élevée'),
        (0.6, '#ffbb33', 'Élevée'),
        (0.4, '#ff8800', 'Modérée'),
        (0.2, '#ff4444', 'Faible'),
    ]
    for y, color, label in seuils:
        fig.add_hline(
            y=y,
            line_dash='dot',
            line_color=color,
            line_width=1,
            opacity=0.5,
            annotation_text=label,
            annotation_position='right',
            annotation_font=dict(
                color=color, size=10),
            row=1, col=1)

    # ── Annotations min/max ──────────────────
    idx_min = fci_series.idxmin()
    idx_max = fci_series.idxmax()

    for idx, val, color, label in [
        (idx_min, fci_series.min(),
         '#ff4444', f'Min {fci_series.min():.3f}'),
        (idx_max, fci_series.max(),
         '#00C851', f'Max {fci_series.max():.3f}'),
    ]:
        fig.add_annotation(
            x=idx, y=val,
            text=label,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=color,
            font=dict(color=color, size=10),
            bgcolor='#0e1117',
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
            row=1, col=1)

    # ── VIX ──────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=vix_series.index,
            y=vix_series.values,
            mode='lines',
            name='VIX',
            line=dict(
                color='#f0883e',
                width=1.5),
            fill='tozeroy',
            fillcolor='rgba(240,136,62,0.08)',
            hovertemplate=(
                '<b>%{x|%d %b %Y}</b><br>'
                'VIX : <b>%{y:.1f}</b>'
                '<extra></extra>')),
        row=2, col=1)

    for y, color in [(20, '#00C851'),
                      (30, '#ff4444')]:
        fig.add_hline(
            y=y,
            line_dash='dot',
            line_color=color,
            line_width=1,
            opacity=0.5,
            row=2, col=1)

    # ── Layout ───────────────────────────────
    fig.update_layout(
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(
            color='#c9d1d9',
            family='Inter, sans-serif'),
        legend=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            borderwidth=1,
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1),
        margin=dict(l=0, r=80,
                    t=10, b=0),
        height=440,
        hovermode='x unified',
        xaxis=dict(
            gridcolor='#21262d',
            showgrid=True,
            zeroline=False),
        yaxis=dict(
            gridcolor='#21262d',
            showgrid=True,
            range=[0, 1.05],
            title='FCI',
            zeroline=False),
        xaxis2=dict(
            gridcolor='#21262d',
            showgrid=True,
            zeroline=False),
        yaxis2=dict(
            gridcolor='#21262d',
            showgrid=True,
            title='VIX',
            zeroline=False),
    )

    st.plotly_chart(
        fig, use_container_width=True)


def render_overview(
        data: pd.DataFrame,
        df_fci: pd.DataFrame,
        stats: dict) -> None:
    """
    Rendu complet de la page Vue générale

    Parameters
    ----------
    data   : pd.DataFrame — données brutes
    df_fci : pd.DataFrame — FCI rolling
    stats  : dict — statistiques descriptives
    """
    fci_series  = df_fci['FCI']
    vix_aligned = data['vix'].reindex(
        fci_series.index)

    # Score card
    render_score_card(fci_series)

    st.markdown("<br>", unsafe_allow_html=True)

    # Graphique
    st.markdown("""
    <div style='
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #30363d;'>
        Évolution du FCI — Rolling 252 jours
    </div>
    """, unsafe_allow_html=True)

    render_fci_chart(fci_series, vix_aligned)

    # Infos marché
    st.markdown("""
    <div style='
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #30363d;'>
        Informations marché
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    infos = [
        (c1, str(stats['n_obs']),
         'Observations', '#58a6ff'),
        (c2, str(stats['corr']),
         'Corrélation Spot-Futures', '#56d364'),
        (c3, str(stats['vix_mean']),
         'VIX Moyen', '#f0883e'),
        (c4, f"{stats['start']} → {stats['end']}",
         'Période', '#bc8cff'),
    ]

    for col, val, label, color in infos:
        with col:
            st.markdown(f"""
            <div style='
                background-color: #161b22;
                border: 1px solid #30363d;
                border-left: 3px solid {color};
                border-radius: 8px;
                padding: 12px 16px;'>
                <div style='
                    font-size: 1.1rem;
                    font-weight: 600;
                    color: {color};'>
                    {val}
                </div>
                <div style='
                    font-size: 0.75rem;
                    color: #8b949e;
                    margin-top: 4px;'>
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)
