# ── components/data_view.py ──────────────────────────
# Page 4 — Données brutes et export

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.tests import run_all_tests


def render_market_info(
        data: pd.DataFrame,
        stats: dict) -> None:
    """
    Informations générales sur le marché
    """
    cols = st.columns(4, gap='small')

    infos = [
        ('📊', str(stats['n_obs']),
         'Observations',
         '#58a6ff'),
        ('📈', f"{stats['spot_last']:,.0f}",
         'S&P 500 dernier prix',
         '#56d364'),
        ('🌡️', f"{stats['vix_mean']:.1f}",
         'VIX moyen',
         '#f0883e'),
        ('🔗', f"{stats['corr']:.4f}",
         'Corrélation Spot-Futures',
         '#bc8cff'),
    ]

    for col, (icon, val, label, color) \
            in zip(cols, infos):
        with col:
            st.markdown(f"""
            <div style='
                background-color: #161b22;
                border: 1px solid #30363d;
                border-left: 3px solid {color};
                border-radius: 10px;
                padding: 14px 16px;'>
                <div style='
                    font-size: 1.3rem;
                    margin-bottom: 6px;'>
                    {icon}
                </div>
                <div style='
                    font-size: 1.4rem;
                    font-weight: 700;
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


def render_price_chart(
        data: pd.DataFrame) -> None:
    """
    Graphique prix spot et futures
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['spot'],
        mode='lines',
        name='S&P 500 Spot',
        line=dict(
            color='#58a6ff',
            width=1.5),
        hovertemplate=(
            '<b>Spot</b><br>'
            '%{x|%d %b %Y}<br>'
            'Prix : <b>%{y:,.0f}</b>'
            '<extra></extra>')))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['futures'],
        mode='lines',
        name='E-mini Futures',
        line=dict(
            color='#56d364',
            width=1.5,
            dash='dot'),
        hovertemplate=(
            '<b>Futures</b><br>'
            '%{x|%d %b %Y}<br>'
            'Prix : <b>%{y:,.0f}</b>'
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
        margin=dict(l=0, r=0, t=30, b=0),
        height=300,
        hovermode='x unified',
        xaxis=dict(gridcolor='#21262d'),
        yaxis=dict(
            gridcolor='#21262d',
            title='Prix'),
    )

    st.plotly_chart(
        fig, use_container_width=True)


def render_tests_results(
        data: pd.DataFrame) -> None:
    """
    Résultats des tests économétriques
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
        Tests économétriques préalables
    </div>
    """, unsafe_allow_html=True)

    with st.spinner('Calcul des tests...'):
        tests = run_all_tests(data)

    # ADF
    st.markdown("""
    <div style='font-size:0.85rem;
                color:#8b949e;
                margin-bottom:8px;'>
        Test ADF — Stationnarité
    </div>
    """, unsafe_allow_html=True)

    adf_results = [
        tests['adf']['spot_level'],
        tests['adf']['fut_level'],
        tests['adf']['spot_diff'],
        tests['adf']['fut_diff'],
    ]

    cols_adf = st.columns(4, gap='small')
    for col, r in zip(cols_adf, adf_results):
        ok    = r['stationnaire']
        color = '#00C851' if ok else '#ff4444'
        icon  = '✅' if ok else '❌'
        with col:
            st.markdown(f"""
            <div style='
                background-color: #161b22;
                border: 1px solid #30363d;
                border-left: 3px solid {color};
                border-radius: 8px;
                padding: 10px 12px;
                margin-bottom: 6px;'>
                <div style='
                    font-size: 0.75rem;
                    color: #8b949e;
                    margin-bottom: 4px;'>
                    {r['name']}
                </div>
                <div style='
                    font-size: 0.85rem;
                    font-weight: 600;
                    color: {color};'>
                    {icon}
                    {'Stationnaire'
                     if ok
                     else 'Non stationnaire'}
                </div>
                <div style='
                    font-size: 0.75rem;
                    color: #6e7681;
                    margin-top: 4px;'>
                    stat = {r['stat']:.4f}<br>
                    p = {r['pvalue']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Johansen + Granger
    col_j, col_g = st.columns(2, gap='large')

    with col_j:
        j = tests['johansen']
        ok_j  = j.get('coint_confirmed', False)
        color = '#00C851' if ok_j else '#ff4444'
        icon  = '✅' if ok_j else '❌'

        st.markdown(f"""
        <div style='
            background-color: #161b22;
            border: 1px solid #30363d;
            border-left: 3px solid {color};
            border-radius: 8px;
            padding: 14px 16px;
            margin-top: 8px;'>
            <div style='
                font-size: 0.8rem;
                color: #8b949e;
                margin-bottom: 6px;'>
                Test de Johansen —
                Cointégration
            </div>
            <div style='
                font-size: 1rem;
                font-weight: 600;
                color: {color};'>
                {icon}
                {'Cointégration confirmée'
                 if ok_j
                 else 'Non cointégré'}
            </div>
            <div style='
                font-size: 0.8rem;
                color: #6e7681;
                margin-top: 8px;'>
                Trace = {j.get(
                    'trace_stat', 'N/A')}<br>
                Critique 5% = {j.get(
                    'crit_5pct', 'N/A')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_g:
        g     = tests['granger']
        color = ('#00C851'
                 if g['score'] > 0.5
                 else '#ffbb33')

        st.markdown(f"""
        <div style='
            background-color: #161b22;
            border: 1px solid #30363d;
            border-left: 3px solid {color};
            border-radius: 8px;
            padding: 14px 16px;
            margin-top: 8px;'>
            <div style='
                font-size: 0.8rem;
                color: #8b949e;
                margin-bottom: 6px;'>
                Test de Granger —
                Causalité
            </div>
            <div style='
                font-size: 1rem;
                font-weight: 600;
                color: {color};'>
                {g['direction']}
            </div>
            <div style='
                font-size: 0.8rem;
                color: #6e7681;
                margin-top: 8px;'>
                F(Fut→Spot) =
                {g['f_f2s_mean']:.4f}<br>
                F(Spot→Fut) =
                {g['f_s2f_mean']:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Résumé global
    all_valid = tests['all_valid']
    color_v   = '#00C851' if all_valid \
                else '#ff4444'
    icon_v    = '✅' if all_valid else '❌'

    st.markdown(f"""
    <div style='
        background-color: #161b22;
        border: 1px solid {color_v};
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 12px;
        text-align: center;'>
        <span style='
            font-size: 0.9rem;
            font-weight: 600;
            color: {color_v};'>
            {icon_v} Conditions FCI :
            {'Toutes vérifiées — FCI valide'
             if all_valid
             else 'Conditions non satisfaites'}
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_data_table(
        data: pd.DataFrame,
        df_fci: pd.DataFrame,
        start_date: str,
        end_date: str) -> None:
    """
    Tableau des données et export CSV
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
        Données FCI Rolling
    </div>
    """, unsafe_allow_html=True)

    # Tableau
    df_display = df_fci[
        ['FCI', 'IS', 'VECM',
         'Granger', 'VIX']].copy()
    df_display.index = \
        df_display.index.strftime('%Y-%m-%d')
    df_display = df_display\
        .sort_index(ascending=False)\
        .round(4)

    st.dataframe(
        df_display,
        use_container_width=True,
        height=350)

    # Export
    col_dl1, col_dl2, _ = st.columns(
        [1, 1, 2])

    with col_dl1:
        csv_fci = df_display.to_csv()
        st.download_button(
            label="⬇️ FCI CSV",
            data=csv_fci,
            file_name=(
                f"FCI_SP500_"
                f"{start_date}_"
                f"{end_date}.csv"),
            mime='text/csv',
            use_container_width=True)

    with col_dl2:
        data_export = data[
            ['spot', 'futures',
             'vix', 'ret_spot',
             'ret_futures']].copy()
        data_export.index = \
            data_export.index.strftime(
                '%Y-%m-%d')
        csv_data = data_export\
            .round(4).to_csv()
        st.download_button(
            label="⬇️ Données CSV",
            data=csv_data,
            file_name=(
                f"SP500_data_"
                f"{start_date}_"
                f"{end_date}.csv"),
            mime='text/csv',
            use_container_width=True)


def render_data_view(
        data: pd.DataFrame,
        df_fci: pd.DataFrame,
        stats: dict,
        start_date: str,
        end_date: str) -> None:
    """
    Rendu complet de la page Données

    Parameters
    ----------
    data       : pd.DataFrame
    df_fci     : pd.DataFrame — FCI rolling
    stats      : dict
    start_date : str
    end_date   : str
    """

    # Infos marché
    st.markdown("""
    <div style='
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #30363d;'>
        Informations marché
    </div>
    """, unsafe_allow_html=True)

    render_market_info(data, stats)

    st.markdown(
        "<br>", unsafe_allow_html=True)

    # Prix
    st.markdown("""
    <div style='
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #30363d;'>
        Prix Spot vs Futures
    </div>
    """, unsafe_allow_html=True)

    render_price_chart(data)

    # Tests
    render_tests_results(data)

    # Tableau + export
    render_data_table(
        data, df_fci,
        start_date, end_date)
