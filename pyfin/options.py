"""
pyfin.options

Simple option pricing functions.


Conventions:
- option_type can be "call", "put", "c", or "p".
- exercise can be "european" or "american".
- r and dividend_yield are continuously compounded annual rates.
- time to maturity is measured in years.
- volatility is annualised.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _normalise_option_type(option_type):
    """Convert option type aliases to 'call' or 'put'."""
    option_type = option_type.lower()
    if option_type in ["c", "call"]:
        return "call"
    if option_type in ["p", "put"]:
        return "put"
    raise ValueError("option_type must be 'call', 'put', 'c', or 'p'.")


def _normalise_exercise(exercise):
    """Convert exercise style to lower-case standard form."""
    exercise = exercise.lower()
    if exercise in ["european", "american"]:
        return exercise
    raise ValueError("exercise must be 'european' or 'american'.")



def time_to_maturity(price_date, expiry_date, day_count=365):
    """
    Compute time to maturity in years.

    Parameters
    ----------
    price_date : str, datetime-like, or pandas Series
        Option price date.
    expiry_date : str, datetime-like, or pandas Series
        Option expiry date.
    day_count : int or float, default 365
        Day-count denominator.

    Returns
    -------
    float or pandas Series
        Time to maturity in years.
    """
    # Delayed pandas import keeps the dependency local to this helper.
    import pandas as pd

    price_date = pd.to_datetime(price_date)
    expiry_date = pd.to_datetime(expiry_date)
    return (expiry_date - price_date).dt.days / day_count if hasattr(expiry_date - price_date, "dt") else (expiry_date - price_date).days / day_count


# ---------------------------------------------------------------------
# Utility function
# ---------------------------------------------------------------------

def present_value(cashflows, timeline, rate):
    """
    Present value of cash flows using continuous compounding.

    Parameters
    ----------
    cashflows : array-like
        Cash flow amounts.
    timeline : array-like
        Cash flow timings in years.
    rate : float or array-like
        Continuously compounded discount rate.

    Returns
    -------
    float
        Present value.
    """
    cashflows = np.asarray(cashflows, dtype=float)
    timeline = np.asarray(timeline, dtype=float)
    return np.sum(cashflows * np.exp(-rate * timeline))


# ---------------------------------------------------------------------
# Black-Scholes-Merton pricer
# ---------------------------------------------------------------------

def black_scholes_price(S0, K, r, dividend_yield, sigma, T, option_type):
    """
    Price European vanilla options using the Black-Scholes-Merton model.

    Parameters
    ----------
    S0 : float or array-like
        Spot price.
    K : float or array-like
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Continuous dividend yield.
    sigma : float or array-like
        Annualised volatility.
    T : float or array-like
        Time to maturity in years.
    option_type : str
        "call", "put", "c", or "p".

    Returns
    -------
    float or ndarray
        Option value(s).
    """
    option_type = _normalise_option_type(option_type)

    S0 = np.asarray(S0, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)

    if np.any(S0 <= 0):
        raise ValueError("S0 must be positive.")
    if np.any(K <= 0):
        raise ValueError("K must be positive.")
    if np.any(T < 0):
        raise ValueError("T must be non-negative.")
    if np.any(sigma < 0):
        raise ValueError("sigma must be non-negative.")

    # Handle expiry separately to avoid divide-by-zero warnings.
    if np.all(T == 0):
        if option_type == "call":
            return np.maximum(S0 - K, 0)
        return np.maximum(K - S0, 0)

    # Handle zero volatility as a deterministic forward payoff.
    if np.all(sigma == 0):
        forward = S0 * np.exp((r - dividend_yield) * T)
        if option_type == "call":
            return np.exp(-r * T) * np.maximum(forward - K, 0)
        return np.exp(-r * T) * np.maximum(K - forward, 0)

    d1 = (np.log(S0 / K) + (r - dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S0 * np.exp(-dividend_yield * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-dividend_yield * T) * norm.cdf(-d1)

    return price


# ---------------------------------------------------------------------
# BSM implied volatility
# ---------------------------------------------------------------------

def black_scholes_implied_vol(
    S0,
    K,
    r,
    dividend_yield,
    T,
    option_type,
    market_price,
    lower=1e-6,
    upper=5.0,
):
    """
    Compute implied volatility for a European option using BSM.

    Parameters
    ----------
    S0, K, r, dividend_yield, T, option_type : see black_scholes_price
    market_price : float
        Observed option price.
    lower : float
        Lower volatility bound.
    upper : float
        Upper volatility bound.

    Returns
    -------
    float
        Implied volatility. Returns np.nan if no solution is bracketed.
    """
    def error(sigma):
        return black_scholes_price(S0, K, r, dividend_yield, sigma, T, option_type) - market_price

    try:
        return brentq(error, lower, upper)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------
# Binomial pricer
# ---------------------------------------------------------------------

def binomial_price(
    S0,
    K,
    r,
    sigma,
    T,
    option_type,
    steps,
    dividend_yield=None,
    dividends=None,
    yted=None,
    exercise="american",
):
    """
    Price vanilla options using a Cox-Ross-Rubinstein binomial tree.

    This is based on binomial_pricer.m, with one addition:
    exercise can be "american" or "european".

    Parameters
    ----------
    S0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility.
    T : float
        Time to maturity in years.
    option_type : str
        "call", "put", "c", or "p".
    steps : int
        Number of binomial steps.
    dividend_yield : float, optional
        Continuous dividend yield. Use this OR discrete dividends, not both.
    dividends : array-like, optional
        Discrete dividend amounts.
    yted : array-like, optional
        Years to Ex-dividend dates.
    exercise : str, default "american"
        "american" or "european".

    Returns
    -------
    float
        Option value.
    """
    option_type = _normalise_option_type(option_type)
    exercise = _normalise_exercise(exercise)

    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if dividend_yield is not None and dividends is not None:
        raise ValueError("Use either continuous dividend_yield or discrete dividends, not both.")

    if T == 0:
        if option_type == "call":
            return max(S0 - K, 0)
        return max(K - S0, 0)

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt)) if sigma > 0 else 1.0
    d = 1 / u if sigma > 0 else 1.0
    df = np.exp(-r * dt)

    y_used = 0.0 if dividend_yield is None else dividend_yield

    if sigma == 0:
        forward = S0 * np.exp((r - y_used) * T)
        if option_type == "call":
            value = np.exp(-r * T) * max(forward - K, 0)
        else:
            value = np.exp(-r * T) * max(K - forward, 0)
        if exercise == "american":
            if option_type == "call":
                value = max(value, S0 - K)
            else:
                value = max(value, K - S0)
        return value

    q = (np.exp((r - y_used) * dt) - d) / (u - d)

    if q < 0 or q > 1:
        raise ValueError("Risk-neutral probability is outside [0, 1]. Check inputs.")

    # -----------------------------------------------------------------
    # Build stock price tree
    # -----------------------------------------------------------------
    St = np.zeros((steps + 1, steps + 1))
    ft = np.zeros((steps + 1, steps + 1))

    # No discrete dividends: standard CRR tree.
    if dividends is None:
        St[0, 0] = S0
        for j in range(1, steps + 1):
            for i in range(j):
                St[i, j] = St[i, j - 1] * u
            St[j, j] = St[j - 1, j - 1] * d

    # Discrete dividend case: subtract PV of dividends, build ex-dividend tree,
    # then add back PV of future dividends before each ex-dividend date.
    else:
        if yted is None:
            raise ValueError("yted must be supplied with dividends.")

        dividends = np.asarray(dividends, dtype=float)
        yted = np.asarray(yted, dtype=float)

        if len(dividends) != len(yted):
            raise ValueError("dividends and yted must have the same length.")

        pv_dividends = present_value(dividends, yted, r)
        St[0, 0] = S0 - pv_dividends

        for j in range(1, steps + 1):
            for i in range(j):
                St[i, j] = St[i, j - 1] * u
            St[j, j] = St[j - 1, j - 1] * d

        tau = 0.0
        for j in range(steps):
            time_to_xdate = yted - tau
            future = time_to_xdate > 0
            if np.any(future):
                div_to_add_back = present_value(dividends[future], time_to_xdate[future], r)
                St[: j + 1, j] = St[: j + 1, j] + div_to_add_back
            tau += dt

    # -----------------------------------------------------------------
    # Terminal payoff
    # -----------------------------------------------------------------
    if option_type == "call":
        ft[:, -1] = np.maximum(St[:, -1] - K, 0)
    else:
        ft[:, -1] = np.maximum(K - St[:, -1], 0)

    # -----------------------------------------------------------------
    # Backward induction
    # -----------------------------------------------------------------
    for j in range(steps - 1, -1, -1):
        for i in range(j + 1):
            continuation_value = df * (q * ft[i, j + 1] + (1 - q) * ft[i + 1, j + 1])

            if exercise == "american":
                if option_type == "call":
                    exercise_value = St[i, j] - K
                else:
                    exercise_value = K - St[i, j]
                ft[i, j] = max(continuation_value, exercise_value)
            else:
                ft[i, j] = continuation_value

    return ft[0, 0]


# ---------------------------------------------------------------------
# Binomial implied volatility
# ---------------------------------------------------------------------

def binomial_implied_vol(
    S0,
    K,
    r,
    T,
    option_type,
    steps,
    market_price,
    dividend_yield=None,
    dividends=None,
    yted=None,
    exercise="american",
    lower=0.001,
    upper=5.0,
    grid_size=200,
):
    """
    Compute implied volatility using the binomial model.

    This version is more robust than directly calling brentq(lower, upper),
    because very low volatility can make the CRR risk-neutral probability
    invalid. It searches for a valid bracket before applying brentq.

    Returns
    -------
    float
        Implied volatility. Returns np.nan if no valid solution is found.
    """

    if market_price <= 0:
        return np.nan

    def model_price(sigma):
        return binomial_price(
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            option_type=option_type,
            steps=steps,
            dividend_yield=dividend_yield,
            dividends=dividends,
            yted=yted,
            exercise=exercise,
        )

    def error(sigma):
        return model_price(sigma) - market_price

    # Search for a valid bracket.
    vol_grid = np.linspace(lower, upper, grid_size)

    prev_sigma = None
    prev_error = None

    for sigma in vol_grid:
        try:
            err = error(sigma)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            continue

        if not np.isfinite(err):
            continue

        if abs(err) < 1e-10:
            return sigma

        if prev_error is not None and prev_error * err < 0:
            try:
                return brentq(error, prev_sigma, sigma)
            except ValueError:
                return np.nan

        prev_sigma = sigma
        prev_error = err

    return np.nan

