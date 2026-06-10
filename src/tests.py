# ── src/tests.py ─────────────────────────────────────
# Tests économétriques préalables
# ADF, Johansen, Granger

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import (
    adfuller, grangercausalitytests)
from statsmodels.tsa.vector_ar.vecm import (
    coint_johansen)


def run_adf(series: pd.Series,
            name: str = '') -> dict:
    """
    Test ADF de stationnarité

    Returns
    -------
    dict :
        stat, pvalue, lags,
        stationnaire, name
    """
    result = adfuller(
        series.dropna(), autolag='AIC')

    return {
        'name'        : name,
        'stat'        : round(result[0], 4),
        'pvalue'      : round(result[1], 4),
        'lags'        : result[2],
        'stationnaire': result[1] < 0.05,
        'crit_1pct'   : round(
            result[4]['1%'], 4),
        'crit_5pct'   : round(
            result[4]['5%'], 4),
    }


def run_johansen(
        log_spot: pd.Series,
        log_futures: pd.Series) -> dict:
    """
    Test de cointégration de Johansen

    Returns
    -------
    dict :
        trace_stat, crit_5pct,
        max_eigen, coint_confirmed
    """
    try:
        jdata  = pd.concat(
            [log_spot, log_futures],
            axis=1).dropna()
        result = coint_johansen(
            jdata, det_order=0, k_ar_diff=2)

        trace    = round(result.lr1[0], 4)
        crit     = round(result.cvt[0, 1], 4)
        max_eig  = round(result.lr2[0], 4)
        crit_max = round(result.cvm[0, 1], 4)

        return {
            'trace_stat'      : trace,
            'crit_5pct'       : crit,
            'max_eigen'       : max_eig,
            'crit_max_5pct'   : crit_max,
            'coint_confirmed' : trace > crit,
            'success'         : True,
        }

    except Exception as e:
        return {
            'coint_confirmed' : False,
            'success'         : False,
            'error'           : str(e),
        }


def run_granger(
        ret_spot: pd.Series,
        ret_futures: pd.Series,
        max_lag: int = 7) -> dict:
    """
    Test de causalité de Granger

    Returns
    -------
    dict :
        f_f2s_mean, f_s2f_mean,
        score, direction,
        results_by_lag
    """
    try:
        gdata = pd.concat(
            [ret_spot, ret_futures],
            axis=1).dropna()
        gdata.columns = [
            'ret_spot', 'ret_futures']

        f_f2s_list = []
        f_s2f_list = []
        p_f2s_list = []
        p_s2f_list = []
        results_by_lag = []

        for lag in range(1, max_lag + 1):
            tf = grangercausalitytests(
                gdata[['ret_spot',
                        'ret_futures']],
                maxlag=lag, verbose=False)
            ts = grangercausalitytests(
                gdata[['ret_futures',
                        'ret_spot']],
                maxlag=lag, verbose=False)

            f_f2s = tf[lag][0][
                'ssr_ftest'][0]
            p_f2s = tf[lag][0][
                'ssr_ftest'][1]
            f_s2f = ts[lag][0][
                'ssr_ftest'][0]
            p_s2f = ts[lag][0][
                'ssr_ftest'][1]

            f_f2s_list.append(f_f2s)
            f_s2f_list.append(f_s2f)
            p_f2s_list.append(p_f2s)
            p_s2f_list.append(p_s2f)

            results_by_lag.append({
                'lag'   : lag,
                'f_f2s' : round(f_f2s, 4),
                'p_f2s' : round(p_f2s, 4),
                'f_s2f' : round(f_s2f, 4),
                'p_s2f' : round(p_s2f, 4),
            })

        f_f2s_mean = np.mean(f_f2s_list)
        f_s2f_mean = np.mean(f_s2f_list)
        total      = f_f2s_mean + f_s2f_mean
        score      = f_f2s_mean / total \
                     if total > 0 else 0.5

        if score > 0.6:
            direction = 'Futures → Spot'
        elif score > 0.4:
            direction = 'Bidirectionnel'
        else:
            direction = 'Spot → Futures'

        return {
            'f_f2s_mean'    : round(
                f_f2s_mean, 4),
            'f_s2f_mean'    : round(
                f_s2f_mean, 4),
            'score'         : round(score, 4),
            'direction'     : direction,
            'results_by_lag': results_by_lag,
            'success'       : True,
        }

    except Exception as e:
        return {
            'score'    : 0.5,
            'direction': 'N/A',
            'success'  : False,
            'error'    : str(e),
        }


def run_all_tests(
        data: pd.DataFrame) -> dict:
    """
    Lance tous les tests préalables
    et retourne un résumé

    Returns
    -------
    dict avec tous les résultats
    """

    # ADF
    adf_spot_level = run_adf(
        data['log_spot'],
        'Log Spot (niveau)')
    adf_fut_level  = run_adf(
        data['log_futures'],
        'Log Futures (niveau)')
    adf_spot_diff  = run_adf(
        data['ret_spot'],
        'Returns Spot')
    adf_fut_diff   = run_adf(
        data['ret_futures'],
        'Returns Futures')

    # I(1) confirmé ?
    i1_confirmed = (
        not adf_spot_level['stationnaire'] and
        not adf_fut_level['stationnaire']  and
        adf_spot_diff['stationnaire']      and
        adf_fut_diff['stationnaire'])

    # Johansen
    johansen = run_johansen(
        data['log_spot'],
        data['log_futures'])

    # Granger
    granger = run_granger(
        data['ret_spot'],
        data['ret_futures'])

    return {
        'adf' : {
            'spot_level' : adf_spot_level,
            'fut_level'  : adf_fut_level,
            'spot_diff'  : adf_spot_diff,
            'fut_diff'   : adf_fut_diff,
            'i1_confirmed': i1_confirmed,
        },
        'johansen' : johansen,
        'granger'  : granger,
        'all_valid': (i1_confirmed and
                      johansen.get(
                          'coint_confirmed',
                          False)),
    }
