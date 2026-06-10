# ── src/regimes.py ───────────────────────────────────
# Détection des régimes de marché
# Méthode 1 : VIX
# Méthode 2 : Markov Switching

import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def get_regime_vix(vix_val: float) -> str:
    """
    Définit le régime de marché selon le VIX

    Parameters
    ----------
    vix_val : float — valeur du VIX

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


def compute_vix_regimes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les régimes VIX sur toute la période

    Returns
    -------
    pd.DataFrame avec colonnes :
        regime_vix, regime_color
    """
    regime_colors = {
        'Stable' : '#00C851',
        'Stress' : '#ffbb33',
        'Crise'  : '#ff4444',
    }

    df = pd.DataFrame(index=data.index)
    df['regime_vix'] = data['vix'].apply(
        get_regime_vix)
    df['regime_color'] = df['regime_vix'].map(
        regime_colors)

    return df


def compute_markov_regimes(
        returns: pd.Series,
        k_regimes: int = 3) -> dict:
    """
    Estime le modèle de Markov Switching

    Parameters
    ----------
    returns   : pd.Series — rendements journaliers
    k_regimes : int — nombre de régimes (défaut 3)

    Returns
    -------
    dict avec :
        - smoothed_probs : probabilités lissées
        - regime_labels  : labels des régimes
        - params         : paramètres du modèle
        - transition     : matrice de transition
        - aic            : critère AIC
    """
    try:
        ret_pct = returns.dropna() * 100

        # Estimation
        mod = MarkovRegression(
            ret_pct,
            k_regimes=k_regimes,
            trend='c',
            switching_variance=True
        )
        res = mod.fit(disp=False)

        # Extraction paramètres
        params = []
        for i in range(k_regimes):
            mu    = float(res.params[f'const[{i}]'])
            sigma = float(np.sqrt(
                res.params[f'sigma2[{i}]']))
            params.append({
                'regime' : i,
                'mu'     : mu,
                'sigma'  : sigma,
            })

        # Trier par volatilité croissante
        params_sorted = sorted(
            params, key=lambda x: x['sigma'])

        # Labels selon ordre de volatilité
        regime_labels = {}
        label_names   = ['Stable', 'Stress', 'Crise']
        label_colors  = ['#00C851', '#ffbb33', '#ff4444']

        for rank, p in enumerate(params_sorted):
            regime_labels[p['regime']] = {
                'name'  : label_names[rank],
                'color' : label_colors[rank],
                'mu'    : p['mu'],
                'sigma' : p['sigma'],
            }

        # Matrice de transition
        trans = res.regime_transition
        k     = res.k_regimes
        transition_data = {}
        for i in range(k):
            p_ii  = float(trans[i, i])
            duree = 1 / (1 - p_ii)
            transition_data[i] = {
                'p_ii' : round(p_ii, 4),
                'duree': round(duree, 1),
            }

        return {
            'smoothed_probs' : res.smoothed_marginal_probabilities,
            'regime_labels'  : regime_labels,
            'params'         : params_sorted,
            'transition'     : transition_data,
            'aic'            : res.aic,
            'bic'            : res.bic,
            'k_regimes'      : k_regimes,
            'success'        : True,
        }

    except Exception as e:
        return {
            'success' : False,
            'error'   : str(e),
        }


def get_regime_stats(
        data: pd.DataFrame,
        fci_series: pd.Series) -> pd.DataFrame:
    """
    Calcule les statistiques du FCI par régime

    Parameters
    ----------
    data       : pd.DataFrame — données complètes
    fci_series : pd.Series — FCI rolling

    Returns
    -------
    pd.DataFrame avec stats par régime
    """
    vix_aligned    = data['vix'].reindex(
        fci_series.index)
    regime_aligned = vix_aligned.apply(
        get_regime_vix)

    rows = []
    colors = {
        'Stable' : '#00C851',
        'Stress' : '#ffbb33',
        'Crise'  : '#ff4444',
    }

    for regime in ['Stable', 'Stress', 'Crise']:
        mask    = regime_aligned == regime
        fci_sub = fci_series[mask]
        vix_sub = vix_aligned[mask]

        if len(fci_sub) > 0:
            rows.append({
                'Régime'    : regime,
                'Obs'       : len(fci_sub),
                'FCI moyen' : round(fci_sub.mean(), 4),
                'FCI min'   : round(fci_sub.min(), 4),
                'FCI max'   : round(fci_sub.max(), 4),
                'VIX moyen' : round(vix_sub.mean(), 2),
                'color'     : colors[regime],
            })

    return pd.DataFrame(rows)
