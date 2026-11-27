import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def portfolio_ret(w, mu): 
    """
    Computes the portfolio return as a Scalar given the weights w and the return vector mu.
    
    Inputs
    ===========
    w    : a Series that stores the weights allocated to assets in the portfolio
    mu   : a Series that stores the mean return of each asset in the portfolio
    
    Returns
    ===========
    the portfolio return: as a Scalar 
    """
    
    return w.T @ mu

def portfolio_var(w, omega): 
    """
    Computes the portfolio variance as a Scalar given the weights w and the variance covariance matrix omega.
    
    Inputs
    ===========
    w       : a Series that stores the weights allocated to assets in the portfolio
    omega   : a DataFrame that stores the variance covariance matrix between assets in the portfolio
    
    Returns
    ===========
    the portfolio variance: as a Scalar 
    """
    
    return w.T @ omega @ w

def portfolio_sr(w, mu, omega, rf = 0.03): 
    """
    Computes the portfolio sharpe ratio as a Scalar given the weights w , the variance covariance matrix     omega and risk free rate r
    
    Inputs
    ===========
    w       : a Series that stores the weights allocated to assets in the portfolio
    mu   : a Series that stores the mean return of each asset in the portfolio
    omega   : a DataFrame that stores the variance covariance matrix between assets in the portfolio
    rf         : risk-free rate per annum as a float, defaulted to be 0.03
    
    Returns
    ===========
    the portfolio SR: as a Scalar 
    """
    excess_port_ret = portfolio_ret(w, mu) -rf
    port_std = portfolio_var(w, omega)**0.5
    
    return excess_port_ret/port_std
def msr_optimizer(mu, omega, rf = 0.03):
    """
    Returns the weights of the maximum sharpe ratio portfolio
   
    Inputs
    ===========
    mu         : a Series that stores the mean return of each asset in the portfolio
    omega      : a DataFrame that stores the variance covariance matrix between assets in the portfolio
    rf         : risk-free rate per annum as a float, defaulted to be 0.03
    
    Returns
    ===========
    weights    : portfolio weights, as an ndarray 
    """
    
    # specify constraints
    cons_weights = {'type': 'eq',
                    'fun': lambda w: np.sum(w) - 1
                   }
    
    n = mu.shape[0]
    bnds = ((0, 1),) * n
    
    # Specify objective functions, guess and call minimize
    args = (mu, omega, rf) 
    guess = np.ones(n)*1/n
    
    obj_fun = lambda w, mu, omega, rf: -portfolio_sr(w, mu, omega, rf)
    
    result = minimize(obj_fun, guess, args = args, method = 'SLSQP', bounds = bnds, constraints = cons_weights, options = {'disp': False} ) 
        
    msr_weights = pd.Series(result.x, mu.index)   
    return msr_weights

def gmv_optimizer(omega):
    """
    Returns the weights of the global minimum variance portfolio given a covariance matrix
   
    Inputs
    ===========
    omega      : a DataFrame that stores the variance covariance matrix between assets in the portfolio
    
    Returns
    ===========
    weights    : portfolio weights, as an ndarray
    """
    
    # specify constraints
    cons_weights = {'type': 'eq',
                    'fun': lambda w: np.sum(w) - 1
                    }
    n = omega.shape[0]
    bnds = ((0, 1),) * n
    
    # Specify objective functions, guess and call minimize
   
    guess = np.ones(n)*1/n
    args = (omega,)
    result = minimize(portfolio_var, guess, args = args, method = 'SLSQP', bounds = bnds, constraints = cons_weights, options = {'disp': False} ) 
    gmv_weights = pd.Series(result.x, omega.index)   
    return gmv_weights

def var_optimizer(target_ret, mu, omega):
    """
    Returns portfolio weights that deliver the target portfolio return while minimising portfolio variance.
   
    Inputs
    ===========
    target_ret    : target portfolio return, as a positive integer or floating-point number
    mu            : a Series that stores the mean return of each asset in the portfolio
    omega         : a DataFrame that stores the variance covariance matrix between assets in the portfolio
   
    Returns
    ===========
    weights       : portfolio weights, as an ndarray 
    """
    
    
    #specify constraints
    cons_weights = {'type': 'eq',
                    'fun': lambda w: np.sum(w) - 1
                    }
    
    cons_target = {'type': 'eq',
                 'args': [mu,],
                 'fun': lambda w, mu: portfolio_ret(w, mu) - target_ret
            }
    cons = (cons_weights, cons_target)
    
    n = omega.shape[1]
    bnds = ((0, 1),) * n
    
    # Specify objective functions, guess and call minimize
    guess = np.ones(n)*1/n
    args = (omega,)
    result = minimize(portfolio_var, guess, args = args, method = 'SLSQP', bounds = bnds, constraints = cons, options = {'disp': False} ) 
    
    return result.x




def plot_efficient_frontier(mu, omega, rf=0.03, msr=False, gmv=False, cml=False, num_of_points=100):
    """
    Plot the efficient frontier for a set of assets, with optional display of
    the GMV portfolio, MSR portfolio, and Capital Market Line.

    Inputs
    ===========
    mu              : pd.Series
        Expected asset returns.
    omega           : pd.DataFrame
        Variance-covariance matrix.
    rf              : float, default 0.03
        Annual risk-free rate.
    msr             : bool, default False
        If True, display the maximum Sharpe ratio portfolio.
    gmv             : bool, default False
        If True, display the global minimum variance portfolio.
    cml             : bool, default False
        If True, display the Capital Market Line.
    num_of_points   : int, default 100
        Number of target return points used to trace the efficient frontier.

    Returns
    ===========
    None
    """

    # Basic checks
    if not isinstance(mu, pd.Series):
        raise TypeError("mu must be a pandas Series.")
    if not isinstance(omega, pd.DataFrame):
        raise TypeError("omega must be a pandas DataFrame.")
    if not omega.index.equals(omega.columns):
        raise ValueError("omega must have matching index and columns.")
    if not mu.index.equals(omega.index):
        raise ValueError("mu index must match omega index/columns.")
    if num_of_points < 2:
        raise ValueError("num_of_points must be at least 2.")

    # Define target return range starting from GMV return
    w_gmv = pd.Series(gmv_optimizer(omega), index=mu.index)
    ret_gmv = portfolio_ret(w_gmv, mu)
    target_rets = np.linspace(ret_gmv, mu.max(), num_of_points)

    # Compute efficient frontier points
    frontier_ret = np.zeros(num_of_points)
    frontier_var = np.zeros(num_of_points)

    for i, target in enumerate(target_rets):
        w = pd.Series(var_optimizer(target, mu, omega), index=mu.index)
        frontier_ret[i] = portfolio_ret(w, mu)
        frontier_var[i] = portfolio_var(w, omega)

    # Plot efficient frontier
    plt.figure(figsize=(8, 6))
    plt.plot(np.sqrt(frontier_var), frontier_ret, lw=2)
    plt.xlabel("Volatility (standard deviation)")
    plt.ylabel("Expected return")
    plt.title("Efficient Frontier")

    # Plot GMV portfolio if requested
    if gmv:
        x_gmv = np.sqrt(portfolio_var(w_gmv, omega))
        y_gmv = portfolio_ret(w_gmv, mu)
        plt.plot(x_gmv, y_gmv, 'r*', markersize=10, label="GMV")
        plt.annotate("GMV", (x_gmv, y_gmv), xytext=(5, 5), textcoords="offset points")

    # Compute MSR portfolio once if needed
    w_msr = None
    if msr or cml:
        w_msr = pd.Series(msr_optimizer(mu, omega, rf), index=mu.index)

    # Plot MSR portfolio if requested
    if msr:
        x_msr = np.sqrt(portfolio_var(w_msr, omega))
        y_msr = portfolio_ret(w_msr, mu)
        plt.plot(x_msr, y_msr, 'b*', markersize=10, label="MSR")
        plt.annotate("MSR", (x_msr, y_msr), xytext=(5, 5), textcoords="offset points")

    # Plot CML if requested
    if cml:
        sr_msr = portfolio_sr(w_msr, mu, omega, rf)
        x = np.linspace(0, np.sqrt(frontier_var).max(), 100)
        y = rf + sr_msr * x
        plt.plot(x, y, linestyle='--', label="CML")

    if gmv or msr or cml:
        plt.legend()

    return

def frontier_points(mu, omega, num_of_points=100):
    w_gmv = gmv_optimizer(omega)
    ret_gmv = portfolio_ret(w_gmv, mu)
    target_rets = np.linspace(ret_gmv, mu.max(), num_of_points)

    frontier_ret = np.zeros(num_of_points)
    frontier_var = np.zeros(num_of_points)

    for i, target in enumerate(target_rets):
        w = pd.Series(var_optimizer(target, mu, omega), index=mu.index)
        frontier_ret[i] = portfolio_ret(w, mu)
        frontier_var[i] = portfolio_var(w, omega)

    return np.sqrt(frontier_var), frontier_ret

def sample_cov(ret):
    """
    Covariance matrix estimator. Generates the sample covariance matrix between assets in the returns data.
    
    Inputs
    ===========
    ret           : returns data as a DataFrame. The returns data can be in any time frame such as daily, monthly, etc...
    
    Returns
    ===========
    sample covariance matrix : a DataFrame that stores the sample covariance matrix between assets in the portfolio
    """
    
    return ret.cov()


def constant_corr_cov(ret):
    """
    Covariance matrix estimator. Generates the constant correlation covariance matrix between assets in the returns data.
    
    Inputs
    ===========
    ret           : returns data as a DataFrame. The returns data can be in any time frame such as daily, monthly, etc...
    
    Returns
    ===========
    constant covariance matrix : a DataFrame that stores the constant correlation covariance matrix between assets in the portfolio    
    """
    
    cor = ret.corr()
    n = cor.shape[0]
    cor_bar = (cor.values.sum() - n)/(n*(n-1))
    x = np.ones((n,n))*cor_bar
    np.fill_diagonal(x,1)
    ccor = pd.DataFrame(x, index = cor.index, columns = cor.columns)
    V = pd.DataFrame(ret.std())
    return (V@V.T)*ccor

