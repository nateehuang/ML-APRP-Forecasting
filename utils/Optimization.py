import pandas as pd
import numpy as np
from utils.Data import DataCollection, DataSeries

from pypfopt import black_litterman, risk_models, expected_returns, EfficientFrontier, objective_functions

# This module includes portfolio construction functions
# input_dc can take historical + forecast data vs historical + real data

# annualized mean returns: returns.mean() * frequency
def calculate_annualized_expected_returns(input_price_df: pd.DataFrame, input_freq: str,
                                            returns_data = True, compounding = False ):
    if not isinstance(input_price_df, pd.DataFrame):
        raise ValueError("Not a valid input_price_df. Type should be a pd.DataFrame.")
    if not isinstance(input_freq, str):
        raise ValueError("Not a valid input_freq, please enter daily/monthly.")
    if input_freq == 'daily':
        frequency = 252
    elif input_freq == 'monthly':
        frequency = 12

    res = expected_returns.mean_historical_return(input_price_df, returns_data=returns_data, 
                                        compounding=compounding, frequency=frequency) 
    return res
    
# annualized return covariance matrix: returns.cov() * frequency
def calculate_annualized_return_covariance(input_price_df: pd.DataFrame, input_freq: str,
                                            returns_data = True, compounding = False ):
    if not isinstance(input_price_df, pd.DataFrame):
        raise ValueError("Not a valid input_price_df. Type should be a pd.DataFrame.")
    if not isinstance(input_freq, str):
        raise ValueError("Not a valid input_freq, please enter daily/monthly.")
    if input_freq == 'daily':
        frequency = 252
    elif input_freq == 'monthly':
        frequency = 12

    res = risk_models.sample_cov(input_price_df, returns_data=returns_data, frequency=frequency)
    return res

# Portfolios:
def equal_portfolio(input_price_df: pd.DataFrame):
    ''' 
    Equally weighted Portfolio.
    
    ArgsS
    ----------
        input_price_df: pd.DataFrame
            the time series dataset used
    Returns
    ----------
        weights: 1xn pd.DataFrame
    '''
    if not isinstance(input_price_df, pd.DataFrame):
        raise ValueError("Not a valid input_price_df. Type should be a pd.DataFrame.")
    n = len(input_price_df.columns)
    weights = [[1 / n] * n]
    date = input_price_df.index[-1]
    weights_df = pd.DataFrame(weights, columns=input_price_df.columns, index=[date])
    return weights_df

def portfolio_opt(input_return_df: pd.DataFrame, input_freq: str, solution: str, weight_bounds = (0,1), risk_aversion=1, market_neutral=False,
                                risk_free_rate=0.0, target_volatility=0.01, target_return=0.11, 
                                returns_data=True, compounding=False):
    """
    pyportfolioopt Portfolios

    Args
    ----------
    input_return_df: pd.DataFrame
        historical prices only, or historical + forecasts
    solution: str
        'max_sharpe','min_volatility','max_quadratic_utility',
        'efficient_risk','efficient_return','custom'
    input_freq: str
        daily/monthly

    Returns
    ----------
    weights_df: 1 x n pd.DataFrame

    """ 
    if not isinstance(input_return_df, pd.DataFrame):
        raise ValueError("Not a valid input_price_df. Type should be a pd.DataFrame.") 
    if not isinstance(input_freq, str):
        raise ValueError("Not a valid input_freq, please enter daily/monthly.")
    if not isinstance(solution, str):
        raise ValueError("Not a valid solution.")
    # annualized mean returns: returns.mean() * frequency
    mu = calculate_annualized_expected_returns(input_price_df = input_return_df, 
                                                                input_freq = input_freq, 
                                                                returns_data = returns_data, 
                                                                compounding = compounding)
    # annulized return covariance matrix: returns.cov() * frequency
    S = calculate_annualized_return_covariance(input_price_df = input_return_df,
                                                    input_freq = input_freq,
                                                    returns_data = returns_data, 
                                                    compounding = compounding)
    # Optimise for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S, weight_bounds = weight_bounds, gamma = 0)

    if solution == 'max_sharpe':
        raw_weights = ef.max_sharpe(risk_free_rate = risk_free_rate)
    elif solution == 'min_volatility':
        raw_weights = ef.min_volatility()
    elif solution == 'max_quadratic_utility':
        raw_weights = ef.max_quadratic_utility(risk_aversion = risk_aversion, market_neutral = market_neutral)
    elif solution == 'efficient_risk':
        raw_weights = ef.efficient_risk(target_volatility = target_volatility, market_neutral = market_neutral)
    elif solution == 'efficient_return':
        raw_weights = ef.efficient_return(target_return = target_return, market_neutral = market_neutral)
    elif solution == 'custom':
        print('Outside Implement Required.')
        return None

    date = input_return_df.index[-1] # the date on which weights are calculated
    weights_df = pd.DataFrame(raw_weights, columns = raw_weights.keys(), index=[date])
    return weights_df

#################################################################################
'''
def rebalancing(input_return_df: pd.DataFrame, entire_return_df: pd.DataFrame,
                input_freq: str, solution: str, bounds=(0, 1),risk_aversion=1, market_neutral=False,
                risk_free_rate=0.02, target_volatility=0.01, target_return=0.11, 
                returns_data=True, compounding=False, end_date: pd.datetime):
    
    today = input_return_df.index[-1]
    weights_dict = {} # key=datetime value: weight_df
    if today is before end_date:
        next_day_returns = 
        one_day_weights_df = portfolio_opt(input_return_df, input_freq, solution, bounds,risk_aversion, market_neutral,
                                        risk_free_rate, target_volatility, target_return, returns_data, compounding)
        weights_dict[today] = one_day_weights_df

        next_day_return_list = one_day_weights_df * next_day_returns
        input_return_df += next_day_return
        today = entire_return_df.today_to_next_row_index
    else:
        return pd.DataFrame.from_dict(weight_dicts)
'''
    

# BLACK_LITTERMAN:
def black_litterman_portfolio(input_dc: DataCollection, risk_free_rate = 0.02, frequency=252):

    '''
    The Black-Litterman (BL) model [1] takes a Bayesian approach to asset 
    allocation. Specifically, it combines a prior estimate of returns 
    (canonically, the market-implied returns) with views on certain assets,
    to produce a posterior estimate of expected returns. 
    Priors:
    cov_matrix is a NxN sample covariance matrix
    mcaps is a dict of market caps
    market_prices is a series of S&P500 prices  
    Views:
    In the Black-Litterman model, users can either provide absolute or 
    relative views. Absolute views are statements like: “AAPL will return 
    10%” or “XOM will drop 40%”. Relative views, on the other hand, are 
    statements like “GOOG will outperform FB by 3%”. 
    These views must be specified in the vector Q and mapped to the asset 
    universe via the picking matrix P. The picking matrix is more interesting.
    Remember that its role is to link the views (which mention 8 assets) 
    to the universe of 10 assets.
    '''
    pass
    # prices_df - input_dc.to_df()
    # returns_df = input_dc.get_returns().dropna()

    # # Priors
    # market_prices = pd.Series('SPY daily prices with DatatimeIndex')
    # mcaps = None # market caps dict{tickr:cap} or pd.Series
    
    # cov_matrix = returns_df.cov()
    # # risk aversion positive float
    # delta = black_litterman.market_implied_risk_aversion(market_prices, frequency = frequency, risk_free_rate=risk_free_rate)
    # prior = black_litterman.market_implied_prior_returns(mcaps, delta, cov_matrix, risk_free_rate=risk_free_rate)

    # # Absolute Views
    # Q = np.array([-0.20, 0.05, 0.10, 0.15, ...]).reshape(-1, 1)
    # P = np.array(
    #     [
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0.5, 0.5, -0.5, -0.5, 0, 0],
    #     ]
    # )
    # viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.15}

    # # model:
    # bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)
    # # input: cov_matrix, pi=None, absolute_views=None, Q=None, P=None, 
    # #        omega=None, view_confidences=None, tau=0.05, risk_aversion=1, 
    # #        **kwargs

    # rets = bl.bl_returns()
    # ef = EfficientFrontier(rets, cov_matrix, weight_bounds=(0, 1), gamma=0)

    # # ef.max_quadratic_utility ??
    # raw_weights = ef.max_quadratic_utility(risk_aversion=delta, market_neutral=False)

    # # metrics = ef.portfolio_performance(verbose=False,risk_free_rate=risk_free_rate)
    # # weights = list(raw_weights.values())

    # # OR use return-implied weights
    # # bl.bl_weights(delta)
    # # weights = bl.clean_weights()
    # return weights
    
