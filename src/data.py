# ── src/data.py ──────────────────────────────────────
# Téléchargement et préparation des données S&P 500

import yfinance as yf
import pandas as pd
import numpy as np


@staticmethod
def _clean_series(df, col='Close'):
    """Extrait et nettoie une série de prix"""
    return df[col].squeeze()


def load_data(start: str, end: str) -> pd.DataFrame:
    """
    Télécharge et prépare les données S&P 500

    Parameters
    ----------
    start : str — YYYY-MM-DD
    end   : str — YYYY-MM-DD

    Returns
    -------
    pd.DataFrame :
        spot, futures, vix,
        log_spot, log_futures,
        ret_spot, ret_futures
    """

    # Téléchargement
    sp500  = yf.download(
        '^GSPC', start=start, end=end,
        auto_adjust=True, progress=False)
    sp500f = yf.download(
        'ES=F', start=start, end=end,
        auto_adjust=True, progress=False)
    vix    = yf.download(
        '^VIX', start=start, end=end,
        auto_adjust=True, progress=False)

    # Construction DataFrame
    data = pd.DataFrame({
        'spot'    : sp500['Close'].squeeze(),
        'futures' : sp500f['Close'].squeeze(),
        'vix'     : vix['Close'].squeeze(),
    }).dropna()

    # Transformations log
    data['log_spot']    = np.log(data['spot'])
    data['log_futures'] = np.log(data['futures'])
    data['ret_spot']    = data['log_spot'].diff()
    data['ret_futures'] = data['log_futures'].diff()
    data = data.dropna()

    return data


def get_summary_stats(
        data: pd.DataFrame) -> dict:
    """
    Statistiques descriptives pour l'affichage
    """
    return {
        'n_obs'        : len(data),
        'start'        : data.index[0].date(),
        'end'          : data.index[-1].date(),
        'corr'         : round(
            data['spot'].corr(
                data['futures']), 4),
        'vix_mean'     : round(
            data['vix'].mean(), 2),
        'vix_max'      : round(
            data['vix'].max(), 2),
        'vix_max_date' : data[
            'vix'].idxmax().date(),
        'spot_mean'    : round(
            data['spot'].mean(), 2),
        'spot_last'    : round(
            data['spot'].iloc[-1], 2),
        'ret_std'      : round(
            data['ret_spot'].std() * 100, 4),
    }
