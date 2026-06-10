# ── components/regimes.py ────────────────────────────
# Page 3 — Régimes de marché

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.regimes import (
    REGIME_CONFIG,
    get_regime_stats,
    compute_markov_regimes,
    compare_regimes)


def render_regime_cards(
        data: pd.DataFrame,
        fci_series: pd.Series) -> None:
    """
    Cards FCI moyen par régime
    """
    regime_stats = get_regime_stats(
        data, fci_series)

    cols = st.columns(3, gap='small')

    for col, (_, row) in zip(
            cols,
            regime_stats.iterrows()):

        color = row['color']
        emoji = row['Emoji']

        with col:
            st.markdown(f"""
            <div style='
                background-color: #161b22;
                border: 1px solid #30363d;
                border-top: 3px solid {color};
                border-radius: 12px;
                padding: 20px 16px;
                text-align: center;'>

                <div style='
                    font-size: 1.5rem;
                    margin-bottom: 4px;'>
                    {emoji}
                </div>

                <div style='
                    font-size: 0.75rem;
                    color: #8b949e;
                    text-transform: uppercase;
                    letter-spacing: 1.5px;
                    margin-bottom: 8px;'>
                    Régime {row['Régime']}
                </div>

                <div style='
                    font-size: 2.2rem;
                    font-weight: 700;
                    color: {color};
                    line-height: 1;'>
                    {row['FCI moyen']:.3f}
                </div>

                <div style='
                    font-size: 0.75rem;
                    color: #6e7681;
                    margin-top: 8px;'>
                    {row['Obs']} jours
                    ({row['Pct']:.1f}%)
                </div>

                <div style='
                    font-size: 0.75rem;
                    color: #6e7681;
                    margin-top: 4px;'>
                    VIX moy. {row['VIX moyen']:.1f}
                </div>

                <div style='
                    font-size: 0.7rem;
                    color: #6e7681;
                    margin-top: 4px;
                    padding-top: 8px;
                    border-top: 1px solid
                    #30363d;'>
                    [{row['FCI min']:.3f} —
                    {row['FCI max']:.3f}]
                </div>

            </div>
            """, unsafe_allow_html=True)


def render_regime_chart(
        data: pd.DataFrame,
        fci_series: pd.Series) -> None:
    """
    Graphique FCI coloré par régime VIX
    """
    vix_aligned = data['vix'].reindex(
        fci_series.index)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.03)

    # ── FCI coloré par régime ────────────────
    # Zones de régime
    prev_regime = None
    x_start     = None

    for date, vix_val in vix_aligned.items():
        if vix_val < 20:
            regime = 'Stable'
        elif vix_val < 30:
            regime = 'Stress'
        else:
            regime = 'Crise'

        if regime != prev_regime:
            if prev_regime is not None:
                color = REGIME_CONFIG[
                    prev_regime]['color']
                fig.add_vrect(
                    x0=x_start, x1=date,
                    fillcolor=color,
                    opacity=0.08,
                    line_width=0,
                    row=1, col=1)
            x_start     = date
            prev_regime = regime

    # Dernière zone
    if prev_regime is not None:
        color = REGIME_CONFIG[
            prev_regime]['color']
        fig.add_vrect(
            x0=x_start,
            x1=fci_series.index[-1],
            fillcolor=color,
            opacity=0.08,
            line_width=0,
            row=1, col=1)

    # Ligne FCI
    fig.add_trace(
        go.Scatter(
            x=fci_series.index,
            y=fci_series.values,
            mode='lines',
            name='FCI',
            line=dict(
                color='#58a6ff',
                width=2),
            hovertemplate=(
                '<b>%{x|%d %b %Y}</b><br>'
                'FCI : <b>%{y:.3f}</b>'
                '<extra></extra>')),
        row=1, col=1)

    fig.add_hline(
        y=0.5,
        line_dash='dot',
        line_color='#8b949e',
        line_width=1,
        opacity=0.5,
        row=1, col=1)

    # ── VIX ──────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=vix_aligned.index,
            y=vix_aligned.values,
            mode='lines',
            name='VIX',
            line=dict(
                color='#f0883e',
                width=1.5),
            fill='tozeroy',
            fillcolor='rgba(240,136,62,0.1)',
            hovertemplate=(
                '<b>%{x|%d %b %Y}</b><br>'
                'VIX : <b>%{y:.1f}</b>'
                '<extra></extra>')),
        row=2, col=1)

    for y, color, label in [
        (20, '#00C851', 'Stable'),
        (30, '#ff4444', 'Crise'),
    ]:
        fig.add_hline(
            y=y,
            line_dash='dot',
            line_color=color,
            line_width=1,
            opacity=0.6,
            annotation_text=label,
            annotation_position='right',
            annotation_font=dict(
                color=color, size=9),
            row=2, col=1)

    # ── FCI par régime (barres) ──────────────
    regime_stats = get_regime_stats(
        data, fci_series)

    fig.add_trace(
        go.Bar(
            x=regime_stats['Régime'],
            y=regime_stats['FCI moyen'],
            marker_color=regime_stats['color'],
            marker_line_color='#30363d',
            marker_line_width=1,
            text=regime_stats[
                'FCI moyen'].round(3),
            textposition='outside',
            textfont=dict(
                color='#c9d1d9', size=11),
            hovertemplate=(
                '<b>%{x}</b><br>'
                'FCI moyen : <b>%{y:.3f}</b>'
                '<extra></extra>'),
            name='FCI par régime'),
        row=3, col=1)

    # ── Layout ───────────────────────────────
    fig.update_layout(
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(
            color='#c9d1d9',
            family='Inter, sans-serif'),
        showlegend=False,
        margin=dict(l=0, r=60,
                    t=10, b=0),
        height=520,
        hovermode='x unified',
        bargap=0.4,
    )

    for row in range(1, 4):
        fig.update_xaxes(
            gridcolor='#21262d',
            showgrid=True,
            zeroline=False,
            row=row, col=1)

    fig.update_yaxes(
        gridcolor='#21262d',
        showgrid=True,
        zeroline=False,
        range=[0, 1.1],
        title='FCI',
        row=1, col=1)
    fig.update_yaxes(
        gridcolor='#21262d',
        showgrid=True,
        zeroline=False,
        title='VIX',
        row=2, col=1)
    fig.update_yaxes(
        gridcolor='#21262d',
        showgrid=True,
        zeroline=False,
        range=[0, 0.8],
        title='FCI moy.',
        row=3, col=1)

    st.plotly_chart(
        fig, use_container_width=True)


def render_markov_section(
        data: pd.DataFrame) -> None:
    """
    Section Markov Switching
    """
    st.markdown("""
    <div style='
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #30363d;'>
        Modèle de Markov Switching
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(
            'Estimation Markov Switching...'):
        markov = compute_markov_regimes(
            data['ret_spot'], k_regimes=3)

    if not markov.get('success'):
        st.warning(
            f"⚠️ Markov Switching : "
            f"{markov.get('error', 'Erreur')}")
        return

    # Paramètres
    col_params, col_trans = st.columns(
        2, gap='large')

    with col_params:
        st.markdown("""
        <div style='font-size:0.85rem;
                    color:#8b949e;
                    margin-bottom:8px;'>
            Paramètres par régime
        </div>
        """, unsafe_allow_html=True)

        for i, p in enumerate(
                markov['params']):
            label = markov[
                'regime_labels'][
                p['regime']]['name']
            color = markov[
                'regime_labels'][
                p['regime']]['color']
            st.markdown(f"""
            <div style='
                background-color: #161b22;
                border: 1px solid #30363d;
                border-left: 3px solid {color};
                border-radius: 8px;
                padding: 10px 14px;
                margin-bottom: 6px;
                display: flex;
                justify-content: space-between;
                align-items: center;'>
                <span style='
                    font-size: 0.85rem;
                    font-weight: 600;
                    color: {color};'>
                    {label}
                </span>
                <span style='
                    font-size: 0.8rem;
                    color: #8b949e;'>
                    μ = {p['mu']:.4f}% &nbsp;·&nbsp;
                    σ = {p['sigma']:.4f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

    with col_trans:
        st.markdown("""
        <div style='font-size:0.85rem;
                    color:#8b949e;
                    margin-bottom:8px;'>
            Persistance des régimes
        </div>
        """, unsafe_allow_html=True)

        for i, (regime_id, t) in enumerate(
                markov['transition'].items()):
            color = markov[
                'regime_labels'][
                regime_id]['color']
            pct = int(t['p_ii'] * 100)
            st.markdown(f"""
            <div style='
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 10px 14px;
                margin-bottom: 6px;'>
                <div style='
                    display: flex;
                    justify-content:
                    space-between;
                    margin-bottom: 6px;'>
                    <span style='
                        font-size: 0.8rem;
                        color: {color};
                        font-weight: 600;'>
                        {t['label']}
                    </span>
                    <span style='
                        font-size: 0.8rem;
                        color: #8b949e;'>
                        {t['duree']:.1f} jours
                    </span>
                </div>
                <div style='
                    background-color: #21262d;
                    border-radius: 4px;
                    height: 4px;
                    overflow: hidden;'>
                    <div style='
                        background-color: {color};
                        width: {pct}%;
                        height: 100%;'>
                    </div>
                </div>
                <div style='
                    font-size: 0.7rem;
                    color: #6e7681;
                    margin-top: 4px;'>
                    p_ii = {t['p_ii']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Probabilités lissées
    if markov.get('smoothed_probs') is not None:
        probs  = markov['smoothed_probs']
        labels = markov['regime_labels']

        fig = go.Figure()

        colors_map = {
            i: labels[i]['color']
            for i in labels}
        names_map  = {
            i: labels[i]['name']
            for i in labels}

        for i in range(markov['k_regimes']):
            fig.add_trace(go.Scatter(
                x=probs.index,
                y=probs.iloc[:, i],
                mode='lines',
                name=names_map.get(i, f'R{i}'),
                line=dict(
                    color=colors_map.get(
                        i, '#58a6ff'),
                    width=1.5),
                stackgroup='one',
                fillcolor=f'rgba('
                          f'{int(colors_map.get(i,"#58a6ff")[1:3], 16)},'
                          f'{int(colors_map.get(i,"#58a6ff")[3:5], 16)},'
                          f'{int(colors_map.get(i,"#58a6ff")[5:7], 16)},'
                          f'0.3)',
                hovertemplate=(
                    f'<b>{names_map.get(i)}</b><br>'
                    '%{x|%d %b %Y}<br>'
                    'P : <b>%{y:.3f}</b>'
                    '<extra></extra>')))

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
            margin=dict(
                l=0, r=0, t=30, b=0),
            height=280,
            hovermode='x unified',
            xaxis=dict(
                gridcolor='#21262d'),
            yaxis=dict(
                gridcolor='#21262d',
                range=[0, 1],
                title='Probabilité'),
        )

        st.markdown("""
        <div style='
            font-size: 0.75rem;
            color: #8b949e;
            margin: 16px 0 8px 0;'>
            Probabilités lissées par régime
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(
            fig, use_container_width=True)

        # Comparaison VIX vs Markov
        comp = compare_regimes(data, markov)
        if comp.get('success'):
            st.markdown(f"""
            <div style='
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 12px 16px;
                margin-top: 8px;
                display: flex;
                gap: 32px;'>
                <div>
                    <div style='
                        font-size: 0.7rem;
                        color: #8b949e;
                        text-transform: uppercase;
                        letter-spacing: 1px;'>
                        Corrélation VIX / Markov
                    </div>
                    <div style='
                        font-size: 1.3rem;
                        font-weight: 700;
                        color: #58a6ff;
                        margin-top: 4px;'>
                        {comp['correlation']:.4f}
                    </div>
                </div>
                <div>
                    <div style='
                        font-size: 0.7rem;
                        color: #8b949e;
                        text-transform: uppercase;
                        letter-spacing: 1px;'>
                        Taux d'accord
                    </div>
                    <div style='
                        font-size: 1.3rem;
                        font-weight: 700;
                        color: #56d364;
                        margin-top: 4px;'>
                        {comp['agreement_rate']:.1f}%
                    </div>
                </div>
                <div style='
                    align-self: center;
                    font-size: 0.85rem;
                    color: #8b949e;'>
                    ✅ Les deux méthodes
                    convergent
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_regimes(
        data: pd.DataFrame,
        fci_series: pd.Series) -> None:
    """
    Rendu complet de la page Régimes

    Parameters
    ----------
    data       : pd.DataFrame
    fci_series : pd.Series — FCI rolling
    """

    # Cards régimes
    st.markdown("""
    <div style='
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #30363d;'>
        FCI moyen par régime (VIX)
    </div>
    """, unsafe_allow_html=True)

    render_regime_cards(data, fci_series)

    st.markdown(
        "<br>", unsafe_allow_html=True)

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
        FCI et régimes de marché
    </div>
    """, unsafe_allow_html=True)

    render_regime_chart(data, fci_series)

    # Légende
    st.markdown("""
    <div style='
        display: flex;
        gap: 24px;
        margin-top: 8px;'>
        <div style='font-size:0.8rem;
                    color:#8b949e;'>
            🟢 <b style='color:#00C851;'>
            Stable</b> — VIX &lt; 20
        </div>
        <div style='font-size:0.8rem;
                    color:#8b949e;'>
            🟡 <b style='color:#ffbb33;'>
            Stress</b> — 20 ≤ VIX &lt; 30
        </div>
        <div style='font-size:0.8rem;
                    color:#8b949e;'>
            🔴 <b style='color:#ff4444;'>
            Crise</b> — VIX ≥ 30
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section Markov
    render_markov_section(data)
