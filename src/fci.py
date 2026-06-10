# ── src/fci.py ───────────────────────────────────────
# Calcul du Futures Confidence Index (FCI)

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import VECM


def compute_vix_score(vix_series: pd.Series) -> pd.Series:
    """
    Score VIX inversé et normalisé entre
    les percentiles 5 et 95
    """
    vix_min = vix_series.quantile(0.05)
    vix_max = vix_series.quantile(0.95)

    score = 1 - np.clip(
        (vix_series - vix_min) /
        (vix_max - vix_min), 0, 1)

    return score


def compute_fci_window(window_data: pd.DataFrame) -> dict:
    """
    Calcule le FCI sur une fenêtre de données

    Parameters
    ----------
    window_data : pd.DataFrame
        Données sur la fenêtre glissante
        (log_spot, log_futures, ret_spot,
         ret_futures, vix)

    Returns
    -------
    dict avec les scores IS, VECM, Granger,
    VIX et FCI final
    """

    result = {
        'IS'      : 0.5,
        'VECM'    : 0.5,
        'Granger' : 0.5,
        'VIX'     : 0.5,
        'FCI'     : 0.5,
    }

    try:
        # ── Score VIX ────────────────────────────────
        vix_val = window_data['vix'].iloc[-1]
        vix_min = window_data['vix'].quantile(0.05)
        vix_max = window_data['vix'].quantile(0.95)
        vix_s   = float(1 - np.clip(
            (vix_val - vix_min) /
            (vix_max - vix_min), 0, 1))
        result['VIX'] = vix_s

        # ── VECM + IS ────────────────────────────────
        vdata = window_data[
            ['log_spot', 'log_futures']].dropna()

        vm = VECM(vdata, k_ar_diff=2,
                  coint_rank=1,
                  deterministic='ci')
        vr = vm.fit()

        a_s = vr.alpha[0][0]
        a_f = vr.alpha[1][0]
        b   = vr.beta[1][0]

        # VECM score
        dir_ok = (a_s < 0 and a_f > 0)
        sp     = abs(a_s)
        fp     = abs(a_f)
        spd    = sp / (sp + fp) \
                 if (sp + fp) > 0 else 0.5
        bsc    = max(0, min(1, 1 - abs(1 + b)))
        vecm_s = ((1.0 if dir_ok else 0.3) * 0.4
                  + spd * 0.3
                  + bsc * 0.3)
        result['VECM'] = vecm_s

        # IS score
        res   = vr.resid
        sigma = np.cov(res.T)
        w_raw = np.array([-a_f, a_s])
        w_sum = w_raw.sum()

        if abs(w_sum) > 1e-10:
            w       = w_raw / w_sum
            var_eff = (w[0]**2 * sigma[0, 0] +
                      w[1]**2 * sigma[1, 1] +
                      2*w[0]*w[1]*sigma[0, 1])
            if var_eff > 1e-10:
                c_f  = (w[1]**2 * sigma[1, 1] +
                        w[0]*w[1]*sigma[0, 1])
                IS_s = float(np.clip(
                    c_f / var_eff, 0, 1))
                result['IS'] = IS_s

        # ── Granger score ────────────────────────────
        rdata      = window_data[
            ['ret_spot', 'ret_futures']].dropna()
        ff_list    = []
        sf_list    = []

        for lag in range(1, 4):
            tf = grangercausalitytests(
                rdata[['ret_spot', 'ret_futures']],
                maxlag=lag, verbose=False)
            ts = grangercausalitytests(
                rdata[['ret_futures', 'ret_spot']],
                maxlag=lag, verbose=False)
            ff_list.append(
                tf[lag][0]['ssr_ftest'][0])
            sf_list.append(
                ts[lag][0]['ssr_ftest'][0])

        ff_m = np.mean(ff_list)
        sf_m = np.mean(sf_list)
        tot  = ff_m + sf_m
        gr_s = ff_m / tot if tot > 0 else 0.5
        result['Granger'] = gr_s

    except Exception:
        pass

    # ── FCI Final ────────────────────────────────────
    result['FCI'] = (result['IS']      * 0.30 +
                     result['VECM']    * 0.25 +
                     result['Granger'] * 0.15 +
                     result['VIX']     * 0.30)

    return result


def compute_fci_rolling(
        data: pd.DataFrame,
        window: int = 252,
        weights: dict = None) -> pd.DataFrame:
    """
    Calcule le FCI rolling sur toute la période

    Parameters
    ----------
    data    : pd.DataFrame — données complètes
    window  : int — fenêtre glissante (défaut 252)
    weights : dict — poids des composantes
              (défaut IS=0.30, VECM=0.25,
               Granger=0.15, VIX=0.30)

    Returns
    -------
    pd.DataFrame avec colonnes :
        IS, VECM, Granger, VIX, FCI
    """

    if weights is None:
        weights = {
            'IS'      : 0.30,
            'VECM'    : 0.25,
            'Granger' : 0.15,
            'VIX'     : 0.30,
        }

    results = []

    for i in range(window, len(data)):
        w_data = data.iloc[i-window:i].copy()
        date   = data.index[i]

        result = compute_fci_window(w_data)

        # Recalcul FCI avec poids personnalisés
        result['FCI'] = (
            result['IS']      * weights['IS'] +
            result['VECM']    * weights['VECM'] +
            result['Granger'] * weights['Granger'] +
            result['VIX']     * weights['VIX'])

        result['date'] = date
        results.append(result)

    df = pd.DataFrame(results).set_index('date')
    return df


def interpret_fci(score: float) -> dict:
    """
    Retourne l'interprétation du score FCI

    Returns
    -------
    dict avec label, couleur et emoji
    """
    if score >= 0.8:
        return {
            'label'  : 'Très élevée',
            'color'  : '#00C851',
            'emoji'  : '🟢',
            'bg'     : '#0d2b1a'
        }
    elif score >= 0.6:
        return {
            'label'  : 'Élevée',
            'color'  : '#ffbb33',
            'emoji'  : '🟡',
            'bg'     : '#2b2400'
        }
    elif score >= 0.4:
        return {
            'label'  : 'Modérée',
            'color'  : '#ff8800',
            'emoji'  : '🟠',
            'bg'     : '#2b1800'
        }
    elif score >= 0.2:
        return {
            'label'  : 'Faible',
            'color'  : '#ff4444',
            'emoji'  : '🔴',
            'bg'     : '#2b0d0d'
        }
    else:
        return {
            'label'  : 'Très faible',
            'color'  : '#CC0000',
            'emoji'  : '⛔',
            'bg'     : '#1a0000'
        }
