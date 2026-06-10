# ── src/regimes.py ───────────────────────────────────
# Détection des régimes de marché
# Méthode 1 : VIX
# Méthode 2 : Markov Switching

import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import (
    MarkovRegression)


# ── Régimes VIX ──────────────────────────────────────

REGIME_CONFIG = {
    'Stable' : {
        'color'     : '#00C851',
        'bg'        : '#0d2b1a',
        'emoji'     : '🟢',
        'vix_label' : 'VIX < 20',
    },
    'Stress' : {
        'color'     : '#ffbb33',
        'bg'        : '#2b2400',
        'emoji'     : '🟡',
        'vix_label' : '20 ≤ VIX < 30',
    },
    'Crise'  : {
        'color'     : '#ff4444',
        'bg'        : '#2b0d0d',
        'emoji'     : '🔴',
        'vix_label' : 'VIX ≥ 30',
    },
}


def get_regime_vix(vix_val: float) -> str:
    """
    Identifie le régime selon le VIX

    Returns
    -------
    str : 'Stable', 'Stress' ou 'Crise'
    """
    if vix_val < 20:
        return 'Stable'
    elif vix_val < 30:
        return 'Stress'
    else:
        return 'Crise'


def compute_vix_regimes(
        data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les régimes VIX sur toute la période

    Returns
    -------
    pd.DataFrame :
        regime, color, emoji
    """
    df = pd.DataFrame(index=data.index)
    df['regime'] = data['vix'].apply(
        get_regime_vix)
    df['color']  = df['regime'].map(
        {k: v['color']
         for k, v in REGIME_CONFIG.items()})
    df['emoji']  = df['regime'].map(
        {k: v['emoji']
         for k, v in REGIME_CONFIG.items()})
    return df


# ── Markov Switching ─────────────────────────────────

def compute_markov_regimes(
        returns: pd.Series,
        k_regimes: int = 3) -> dict:
    """
    Estime le modèle de Markov Switching

    Parameters
    ----------
    returns   : pd.Series — rendements
    k_regimes : int — nombre de régimes

    Returns
    -------
    dict :
        smoothed_probs, regime_labels,
        params, transition, aic, success
    """
    try:
        ret_pct = returns.dropna() * 100

        mod = MarkovRegression(
            ret_pct,
            k_regimes=k_regimes,
            trend='c',
            switching_variance=True)
        res = mod.fit(disp=False)

        # Paramètres par régime
        params = []
        for i in range(k_regimes):
            mu    = float(
                res.params[f'const[{i}]'])
            sigma = float(np.sqrt(
                res.params[f'sigma2[{i}]']))
            params.append({
                'regime' : i,
                'mu'     : round(mu, 4),
                'sigma'  : round(sigma, 4),
            })

        # Tri par volatilité croissante
        params_sorted = sorted(
            params,
            key=lambda x: x['sigma'])

        # Labels selon ordre de volatilité
        label_names  = [
            'Stable', 'Stress', 'Crise']
        label_colors = [
            '#00C851', '#ffbb33', '#ff4444']
        regime_labels = {}

        for rank, p in enumerate(
                params_sorted):
            regime_labels[p['regime']] = {
                'name'  : label_names[rank],
                'color' : label_colors[rank],
                'mu'    : p['mu'],
                'sigma' : p['sigma'],
            }

        # Matrice de transition
        trans = res.regime_transition
        transition_data = {}
        for i in range(k_regimes):
            p_ii  = float(trans[i, i])
            duree = 1 / (1 - p_ii) \
                    if p_ii < 1 else 999
            transition_data[i] = {
                'p_ii' : round(p_ii, 4),
                'duree': round(duree, 1),
                'label': regime_labels[
                    i]['name'],
            }

        return {
            'smoothed_probs' : res\
                .smoothed_marginal_probabilities,
            'regime_labels'  : regime_labels,
            'params'         : params_sorted,
            'transition'     : transition_data,
            'aic'            : round(
                res.aic, 2),
            'bic'            : round(
                res.bic, 2),
            'k_regimes'      : k_regimes,
            'success'        : True,
        }

    except Exception as e:
        return {
            'success' : False,
            'error'   : str(e),
        }


# ── Stats FCI par régime ─────────────────────────────

def get_regime_stats(
        data: pd.DataFrame,
        fci_series: pd.Series) -> pd.DataFrame:
    """
    Calcule les statistiques FCI par régime VIX

    Parameters
    ----------
    data       : pd.DataFrame
    fci_series : pd.Series — FCI rolling

    Returns
    -------
    pd.DataFrame : stats par régime
    """
    vix_aligned    = data['vix'].reindex(
        fci_series.index)
    regime_aligned = vix_aligned.apply(
        get_regime_vix)

    rows = []
    for regime, config in REGIME_CONFIG.items():
        mask    = regime_aligned == regime
        fci_sub = fci_series[mask]
        vix_sub = vix_aligned[mask]

        if len(fci_sub) > 0:
            rows.append({
                'Régime'    : regime,
                'Emoji'     : config['emoji'],
                'Obs'       : len(fci_sub),
                'Pct'       : round(
                    len(fci_sub) /
                    len(fci_series) * 100, 1),
                'FCI moyen' : round(
                    fci_sub.mean(), 4),
                'FCI min'   : round(
                    fci_sub.min(), 4),
                'FCI max'   : round(
                    fci_sub.max(), 4),
                'FCI std'   : round(
                    fci_sub.std(), 4),
                'VIX moyen' : round(
                    vix_sub.mean(), 2),
                'color'     : config['color'],
                'bg'        : config['bg'],
            })

    return pd.DataFrame(rows)


# ── Corrélation VIX vs Markov ────────────────────────

def compare_regimes(
        data: pd.DataFrame,
        markov_result: dict) -> dict:
    """
    Compare les régimes VIX et Markov

    Returns
    -------
    dict : correlation, agreement_rate
    """
    if not markov_result.get('success'):
        return {'success': False}

    try:
        # Régimes VIX numériques
        vix_num = data['vix'].apply(
            lambda x: 0 if x < 20
            else 1 if x < 30 else 2)

        # Régimes Markov
        probs  = markov_result[
            'smoothed_probs']
        labels = markov_result['regime_labels']

        # Mapper vers ordre stable/stress/crise
        markov_num = pd.Series(
            probs.values.argmax(axis=1),
            index=data.index)

        # Aligner
        common   = vix_num.index.intersection(
            markov_num.index)
        vix_a    = vix_num[common]
        markov_a = markov_num[common]

        corr = float(np.corrcoef(
            vix_a, markov_a)[0, 1])

        agreement = float(
            (vix_a == markov_a).mean())

        return {
            'correlation'   : round(corr, 4),
            'agreement_rate': round(
                agreement * 100, 1),
            'success'       : True,
        }

    except Exception as e:
        return {
            'success': False,
            'error'  : str(e),
        }
