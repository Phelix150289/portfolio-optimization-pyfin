import pandas as pd
import numpy as np

from .portfolio import gmv_optimizer, msr_optimizer


def backtest_strategy(ret, training_window, strategy_rule, **kwargs):
    """
    Backtest a strategy by applying a decision rule through time using rolling windows.

    Parameters
    ----------
    ret : pd.DataFrame
        Asset return data. Rows are time periods and columns are assets.
    training_window : int
        Number of historical periods available to the strategy at each decision point.
    strategy_rule : callable
        Function that maps a historical return window to a vector of portfolio weights
        or positions. The function must return a pd.Series indexed by asset names.
    **kwargs :
        Additional keyword arguments passed to strategy_rule.

    Returns
    -------
    strategy_ret : pd.Series
        Time series of backtested strategy returns.

    Notes
    -----
    This engine assumes:
    1. The strategy is re-estimated each period.
    2. The resulting weights are applied to the next period return only.
    3. Data filtering and modelling choices belong inside strategy_rule.
    """

    if not isinstance(ret, pd.DataFrame):
        raise TypeError("ret must be a pandas DataFrame")

    if not isinstance(training_window, int):
        raise TypeError("training_window must be an integer")

    if training_window <= 0:
        raise ValueError("training_window must be a positive integer")

    if training_window >= len(ret):
        raise ValueError("training_window must be smaller than the number of rows in ret")

    windows = [(start, start + training_window) for start in range(len(ret) - training_window)]

    strategy_weights = []
    decision_dates = ret.index[training_window:]

    for start, end in windows:
        hist_ret = ret.iloc[start:end]
        w = strategy_rule(hist_ret, **kwargs)

        if not isinstance(w, pd.Series):
            raise TypeError("strategy_rule must return a pandas Series")

        # Align returned weights to the full asset universe.
        # Assets not explicitly assigned a weight are set to zero.
        w = w.reindex(ret.columns).fillna(0.0)
        strategy_weights.append(w)

    strategy_weights = pd.DataFrame(strategy_weights, index=decision_dates, columns=ret.columns)

    # The weights are indexed by the first out-of-sample date, so multiplying by ret
    # correctly applies each set of weights to the next-period realised return.
    strategy_ret = (strategy_weights * ret).sum(axis=1, min_count=1)

    return strategy_ret


def strategy_ew(ret):
    """
    Equal-weight strategy.

    Parameters
    ----------
    ret : pd.DataFrame
        Historical return data for the current training window.

    Returns
    -------
    pd.Series
        Equal portfolio weights indexed by asset names.
    """

    if not isinstance(ret, pd.DataFrame):
        raise TypeError("ret must be a pandas DataFrame")

    n_assets = ret.shape[1]
    if n_assets == 0:
        raise ValueError("ret must contain at least one asset")

    return pd.Series(1.0 / n_assets, index=ret.columns)


def strategy_vw(ret, cap_weight):
    """
    Value-weight strategy.

    Uses market-cap weights from the first date in the historical window.

    Parameters
    ----------
    ret : pd.DataFrame
        Historical return data for the current training window.
    cap_weight : pd.DataFrame
        Market-cap weights through time. Rows are dates and columns are assets.

    Returns
    -------
    pd.Series
        Value weights indexed by asset names.
    """

    if not isinstance(ret, pd.DataFrame):
        raise TypeError("ret must be a pandas DataFrame")

    if not isinstance(cap_weight, pd.DataFrame):
        raise TypeError("cap_weight must be a pandas DataFrame")

    if ret.empty:
        raise ValueError("ret must not be empty")

    start_date = ret.index[0]

    if start_date not in cap_weight.index:
        raise KeyError("The start date of the training window is not in cap_weight")

    weights = cap_weight.loc[start_date]

    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, index=cap_weight.columns)

    return weights


def strategy_gmv(ret, cov_method):
    """
    Global minimum variance strategy.

    Parameters
    ----------
    ret : pd.DataFrame
        Historical return data for the current training window.
    cov_method : callable
        Function that estimates a covariance matrix from ret.

    Returns
    -------
    pd.Series
        GMV portfolio weights indexed by asset names.
    """

    if not isinstance(ret, pd.DataFrame):
        raise TypeError("ret must be a pandas DataFrame")

    cov = cov_method(ret)
    weights = gmv_optimizer(cov)

    return pd.Series(weights, index=ret.columns)


def strategy_msr(ret, cov_method, rf=0.03, freq=12):
    """
    Maximum Sharpe ratio strategy.

    Expected returns are estimated using historical sample means from the
    training window.

    Parameters
    ----------
    ret : pd.DataFrame
        Historical return data for the current training window.
    cov_method : callable
        Function that estimates a covariance matrix from ret.
    rf : float, default 0.03
        Annual risk-free rate.
    freq : int, default 12
        Number of return periods per year. For monthly data, use 12.

    Returns
    -------
    pd.Series
        MSR portfolio weights indexed by asset names.
    """

    if not isinstance(ret, pd.DataFrame):
        raise TypeError("ret must be a pandas DataFrame")

    if freq <= 0:
        raise ValueError("freq must be positive")

    # Convert annual risk-free rate to per-period rate
    rf_period = (1 + rf) ** (1 / freq) - 1

    exp_ret = ret.mean()
    cov = cov_method(ret)

    weights = msr_optimizer(exp_ret, cov, rf=rf_period)

    return pd.Series(weights, index=ret.columns)


def strategy_momentum(ret, lookback=12, skip=1, n=5):
    """
    Long-short cross-sectional momentum strategy.

    Parameters
    ----------
    ret : pd.DataFrame
        Historical returns available at the decision point.
    lookback : int, default 12
        Number of periods used to construct the momentum signal.
    skip : int, default 1
        Number of most recent periods skipped when constructing the signal.
    n : int, default 5
        Number of assets held long and short.

    Returns
    -------
    pd.Series
        Portfolio weights indexed by asset names.

    Notes
    -----
    This strategy:
    - ranks assets using cumulative returns over the signal window
    - goes long the top n assets
    - goes short the bottom n assets
    - assigns zero weight to all other assets

    The strategy requires:
        lookback + skip <= len(ret)
    """

    if not isinstance(ret, pd.DataFrame):
        raise TypeError("ret must be a pandas DataFrame")

    if not isinstance(lookback, int) or not isinstance(skip, int) or not isinstance(n, int):
        raise TypeError("lookback, skip, and n must be integers")

    if lookback <= 0:
        raise ValueError("lookback must be positive")

    if skip < 0:
        raise ValueError("skip must be non-negative")

    if n <= 0:
        raise ValueError("n must be positive")

    if lookback + skip > len(ret):
        raise ValueError("lookback + skip cannot exceed available data in the training window")

    if ret.shape[1] == 0:
        raise ValueError("ret must contain at least one asset")

    if skip > 0:
        signal_window = ret.iloc[-(lookback + skip):-skip]
    else:
        signal_window = ret.iloc[-lookback:]

    # Require complete signal history for ranking
    signal_window = signal_window.dropna(axis=1)

    if signal_window.shape[1] < 2 * n:
        raise ValueError("Not enough assets with complete signal history for the chosen n")

    past_ret = (1 + signal_window).prod() - 1
    ranked = past_ret.sort_values(ascending=False)

    winners = ranked.index[:n]
    losers = ranked.index[-n:]

    w = pd.Series(0.0, index=ret.columns)
    w.loc[winners] = 1.0 / n
    w.loc[losers] = -1.0 / n

    return w

