# ── src/fci.py ───────────────────────────────────────
# Calcul du Futures Confidence Index (FCI)

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import (
    grangercausalitytests)
from statsmodels.tsa.vector_ar.vecm import VECM


# ── Interprétation du score ──────────────────────────

def interpret_fci(score: float) -> dict:
    """
    Retourne l'interprétation du score FCI

    Returns
    -------
    dict : label, color, emoji, bg
    """
    if score >= 0.8:
        return {
            'label' : 'Très élevée',
            'color' : '#00C851',
            'emoji' : '🟢',
            'bg'    : '#0d2b1a',
        }
    elif score >= 0.6:
        return {
            'label' : 'Élevée',
            'color' : '#ffbb33',
            'emoji' : '🟡',
            'bg'    : '#2b2400',
        }
    elif score >= 0.4:
        return {
            'label' : 'Modérée',
            'color' : '#ff8800',
            'emoji' : '🟠',
            'bg'    : '#2b1800',
        }
    elif score >= 0.2:
        return {
            'label' : 'Faible',
            'color' : '#ff4444',
            'emoji' : '🔴',
            'bg'    : '#2b0d0d',
        }
    else:
        return {
            'label' : 'Très faible',
            'color' : '#CC0000',
            'emoji' : '⛔',
            'bg'    : '#1a0000',
        }


# ── Score VIX ────────────────────────────────────────

def compute_vix_score(
        vix_val: float,
        vix_min: float,
        vix_max: float) -> float:
    """
    Score VIX inversé et normalisé

    Parameters
    ----------
    vix_val : float — valeur VIX du jour
    vix_min : float — 5e percentile
    vix_max : float — 95e percentile
    """
    return float(1 - np.clip(
        (vix_val - vix_min) /
        (vix_max - vix_min), 0, 1))


# ── Score IS + VECM ──────────────────────────────────

def compute_vecm_is(
        window_data: pd.DataFrame) -> dict:
    """
    Estime le VECM et calcule IS + score VECM

    Returns
    -------
    dict : IS, VECM, alpha_spot,
           alpha_futures, beta
    """
    result = {
        'IS'           : 0.5,
        'VECM'         : 0.5,
        'alpha_spot'   : None,
        'alpha_futures': None,
        'beta'         : None,
        'success'      : False,
    }

    try:
        vdata = window_data[
            ['log_spot',
             'log_futures']].dropna()

        if len(vdata) < 100:
            return result

        vm = VECM(vdata, k_ar_diff=2,
                  coint_rank=1,
                  deterministic='ci')
        vr = vm.fit()

        a_s = vr.alpha[0][0]
        a_f = vr.alpha[1][0]
        b   = vr.beta[1][0]

        # ── Score VECM ───────────────────────
        dir_ok = (a_s < 0 and a_f > 0)
        sp     = abs(a_s)
        fp     = abs(a_f)
        spd    = sp / (sp + fp) \
                 if (sp + fp) > 0 else 0.5
        bsc    = max(0, min(
            1, 1 - abs(1 + b)))
        vecm_s = (
            (1.0 if dir_ok else 0.3) * 0.4 +
            spd * 0.3 +
            bsc * 0.3)

        # ── Score IS ─────────────────────────
        res   = vr.resid
        sigma = np.cov(res.T)
        w_raw = np.array([-a_f, a_s])
        w_sum = w_raw.sum()
        IS_s  = 0.5

        if abs(w_sum) > 1e-10:
            w       = w_raw / w_sum
            var_eff = (
                w[0]**2 * sigma[0, 0] +
                w[1]**2 * sigma[1, 1] +
                2*w[0]*w[1]*sigma[0, 1])
            if var_eff > 1e-10:
                c_f  = (
                    w[1]**2 * sigma[1, 1] +
                    w[0]*w[1]*sigma[0, 1])
                IS_s = float(np.clip(
                    c_f / var_eff, 0, 1))

        result.update({
            'IS'           : IS_s,
            'VECM'         : vecm_s,
            'alpha_spot'   : round(a_s, 6),
            'alpha_futures': round(a_f, 6),
            'beta'         : round(b, 4),
            'success'      : True,
        })

    except Exception:
        pass

    return result


# ── Score Granger ────────────────────────────────────

def compute_granger_score(
        window_data: pd.DataFrame,
        max_lag: int = 3) -> float:
    """
    Calcule le score Granger sur fenêtre

    Returns
    -------
    float : score entre 0 et 1
    """
    try:
        rdata = window_data[
            ['ret_spot',
             'ret_futures']].dropna()

        if len(rdata) < 50:
            return 0.5

        ff_list = []
        sf_list = []

        for lag in range(1, max_lag + 1):
            tf = grangercausalitytests(
                rdata[['ret_spot',
                        'ret_futures']],
                maxlag=lag, verbose=False)
            ts = grangercausalitytests(
                rdata[['ret_futures',
                        'ret_spot']],
                maxlag=lag, verbose=False)
            ff_list.append(
                tf[lag][0]['ssr_ftest'][0])
            sf_list.append(
                ts[lag][0]['ssr_ftest'][0])

        ff_m = np.mean(ff_list)
        sf_m = np.mean(sf_list)
        tot  = ff_m + sf_m

        return ff_m / tot if tot > 0 else 0.5

    except Exception:
        return 0.5


# ── FCI sur une fenêtre ──────────────────────────────

def compute_fci_window(
        window_data: pd.DataFrame,
        vix_min: float,
        vix_max: float,
        weights: dict) -> dict:
    """
    Calcule le FCI sur une fenêtre de données

    Parameters
    ----------
    window_data : pd.DataFrame
    vix_min     : float — 5e percentile global
    vix_max     : float — 95e percentile global
    weights     : dict — poids des composantes

    Returns
    -------
    dict : IS, VECM, Granger, VIX, FCI
    """

    # Score VIX
    vix_val = float(window_data['vix'].iloc[-1])
    vix_s   = compute_vix_score(
        vix_val, vix_min, vix_max)

    # Score VECM + IS
    vecm_is = compute_vecm_is(window_data)

    # Score Granger
    gr_s = compute_granger_score(window_data)

    # Scores
    IS_s   = vecm_is['IS']
    VECM_s = vecm_is['VECM']

    # FCI
    fci = (IS_s   * weights['IS'] +
           VECM_s * weights['VECM'] +
           gr_s   * weights['Granger'] +
           vix_s  * weights['VIX'])

    return {
        'IS'           : IS_s,
        'VECM'         : VECM_s,
        'Granger'      : gr_s,
        'VIX'          : vix_s,
        'FCI'          : float(fci),
        'alpha_spot'   : vecm_is[
            'alpha_spot'],
        'alpha_futures': vecm_is[
            'alpha_futures'],
        'beta'         : vecm_is['beta'],
    }


# ── FCI Rolling ──────────────────────────────────────

def compute_fci_rolling(
        data: pd.DataFrame,
        window: int = 252,
        weights: dict = None) -> pd.DataFrame:
    """
    Calcule le FCI rolling sur toute la période

    Parameters
    ----------
    data    : pd.DataFrame — données complètes
    window  : int — fenêtre glissante
    weights : dict — poids des composantes

    Returns
    -------
    pd.DataFrame : IS, VECM, Granger,
                   VIX, FCI par date
    """

    if weights is None:
        weights = {
            'IS'      : 0.30,
            'VECM'    : 0.25,
            'Granger' : 0.15,
            'VIX'     : 0.30,
        }

    # Bornes VIX globales
    vix_min = float(data['vix'].quantile(0.05))
    vix_max = float(data['vix'].quantile(0.95))

    results = []

    for i in range(window, len(data)):
        w_data = data.iloc[i-window:i].copy()
        date   = data.index[i]

        result = compute_fci_window(
            w_data, vix_min,
            vix_max, weights)
        result['date'] = date
        results.append(result)

    df = pd.DataFrame(results).set_index(
        'date')

    return df
