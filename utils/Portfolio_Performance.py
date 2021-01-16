from utils.Data import DataCollection
from utils.Portfolio import Portfolio
import utils.Optimization as Opt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from scipy.stats import norm

def partial_moment(ret: pd.DataFrame, threshold: float = 0., order: int = 2, lower: bool = True,):
    '''
    Calculate partial moment given return, threshold return, order of the moment and specific part

    Parameters
    ----------
    ret: pd.DataFrame
        return data in dataframe with shape d x 1
    
    threshold: float
        threshold value
    
    order: int
        order of the moment, default = 2
    
    lower: bool
        True if lower partial moment, False if upper, default: True 
    '''
    if ret.shape[1] != 1 or not isinstance(ret, pd.DataFrame):
        raise ValueError("Return should be a d x 1 DataFrame")
    length = ret.shape[0]
    if lower:
        diff_df = threshold - ret
    else:
        diff_df = ret - threshold
    drop_minus = diff_df[diff_df>=0].dropna()
    pm = (drop_minus ** order).sum() / length 
    return pm.item()

def annualized_return(weights: pd.DataFrame, df: pd.DataFrame, freq: str, returns_data = True, compounding = False ):
    '''
    Parameters
    ----------
    df: pd.DataFrame
        price/return data, d x n, date in row, tickers in columns

    Returns
    ----------
    annualized_return: float

    '''
    expected_asset_returns = Opt.calculate_annualized_expected_returns(df, freq, 
                                                                      returns_data = returns_data, 
                                                                      compounding = compounding)
    return np.dot(expected_asset_returns, weights.T).mean()

def annualized_volatility(weights: pd.DataFrame, df: pd.DataFrame, freq: str, returns_data = True, compounding = False):
    '''
    
    '''
    cov = Opt.calculate_annualized_return_covariance(df, freq, returns_data=returns_data, compounding = compounding)
    return np.dot(weights, np.dot(cov, weights.T))[0][0] ** 0.5

def sharpe_ratio(ret, risk, benchmark = 0., ):
    '''
    Calculate risk adjusted return with the form (return - benchmark) / risk
    '''
    return (ret - benchmark) / risk 

def PnL(weights: pd.DataFrame, df: pd.DataFrame, returns_data = True ):
    '''
    portfolio profit/loss on each day
    expected the weights to be a 1 x n df
    '''
    if weights.shape[0] != 1 or not isinstance(weights, pd.DataFrame):
        raise ValueError("weights should be a 1 x n DataFrame")
    if not returns_data: 
        df = df.pct_change().dropna()
    weight_df = pd.DataFrame(weights.values.tolist() * df.shape[0], 
                             columns = df.columns, index = df.index)
    
    pnl = (df * weight_df).sum(axis = 1).to_frame().rename(columns={0:'PnL'})
    return pnl

def max_drawdown(pnl: pd.DataFrame):
    '''
    Parameters
    ----------
    pnl: pd.DataFrame
        a d x 1 df with daily return/PnL of the portfolio
    
    Returns
    ----------
    max_draw: float
        maximum drawdown
    '''
    ret = pd.DataFrame([1], columns = ['PnL'])
    pnl_ret = pnl + 1
    ret = ret.append(pnl_ret)
    price = ret.cumprod()
    max_price = price.cummax()
    max_draw = ((price - max_price) / max_price).min()[0]
    return max_draw 

def drawdowns(pnl: pd.DataFrame):
    pass

def M2():
    pass

class PortfolioPerformance(object):
    '''
    Attributes
    ----------
    portfolio: Portfolio Object
    label: string
    historical_dc: DataCollection
    forecast_dc: DataCollection
    Methods
    ----------
    annualized_return()
    annualized_volatility()
    annualized_sharpe_ratio()
    info_ratio()
    sortino_ratio()
    max_drawdown()
    max_loss()
    expected_shortfall()
    
    '''
   
    def __init__(self, portfolio: Portfolio, evaluate_dc: DataCollection):
        if portfolio.get_tickers() != evaluate_dc.ticker_list():
            raise ValueError("Tickers in portfolio and evaluate data do not match")
        
        self.portfolio = portfolio
        # Check this
        self.label = portfolio.get_solution()
        if portfolio.get_freq() != evaluate_dc.get_freq():
            raise ValueError("The frequency of the data and portfolio do not match")
        self.price_df = evaluate_dc.to_df().dropna()
        self.freq = evaluate_dc.get_freq()
        self.evaluate_dc = evaluate_dc
        self.metrics = {}
    
    def get_metrics(self, metrics: str):
        if metrics in self.metrics:
            return self.metrics[metrics]
        else:
            raise KeyError(metrics + 'is not calculated yet')
    
    def print_metrics(self,):
        '''
        print all metrics in this object
        '''
        for k, v in self.metrics.items():
            print(str(k) + ': ' + str(v))

    def metrics_table(self,):
        '''
        return all available metrics in a pd.DataFrame
        '''
        pass

    # individual metric
    def annualized_return(self, compounding = False ):
        weights = self.portfolio.get_initial_weight()
        df = self.price_df
        freq = self.freq
        self.metrics['annualized_return'] = annualized_return(weights, df, freq, compounding = compounding) 

    def annualized_volatility(self, compounding = False ):
        weights = self.portfolio.get_initial_weight()
        df = self.price_df
        freq = self.freq
        self.metrics['annualized_volatility'] = annualized_volatility(weights, df, freq, compounding = compounding) 

    def annualized_sharpe_ratio(self, risk_free_rate = 0., ):
        if 'annualized_return' not in self.metrics:
            self.annualized_return()
        if 'annualized_volatility' not in self.metrics:
            self.annualized_volatility()
        returns = self.metrics['annualized_return']
        risk = self.metrics['annualized_volatility']
        self.metrics['sharpe_ratio'] = sharpe_ratio(returns, risk, risk_free_rate) 

    def PnL(self, ):
        '''
        portfolio profit/loss on each day with initial weight from the portfolio
        '''
        df = self.price_df
        weight = self.portfolio.get_initial_weight()
        self.metrics['PnL'] = PnL(weight, df)

    def sortino_ratio(self, threshold: float = 0.):
        '''
        Parameters
        ---------
        threshold: float
            benchmark for sortino ratio
        '''
        if 'annualized_return' not in self.metrics:
            self.annualized_return()
        if 'PnL' not in self.metrics:
            self.PnL()
        PnL = self.metrics['PnL']
        expected = self.metrics['annualized_return']
        lpm = partial_moment(PnL, threshold, order = 2, lower = True) ** 0.5
        
        sortino = sharpe_ratio(expected, lpm, threshold)
        self.metrics['sortino_ratio'] = sortino

    def max_drawdown(self, ):
        if 'PnL' not in self.metrics:
            self.PnL()
        self.metrics['max_drawdown'] = max_drawdown(self.metrics['PnL'])

    def omega_ratio(self,  threshold: float = 0., ):
        if 'annualized_return' not in self.metrics:
            self.annualized_return()
        if 'PnL' not in self.metrics:
            self.PnL()

        expected = self.metrics['annualized_return']
        PnL = self.metrics['PnL']
        lpm = partial_moment(PnL, threshold, order = 1, lower = True)
        omega = ((expected - threshold) / lpm ) + 1
        
        self.metrics['omega_ratio'] = omega

    def pain_index(self, ):
        pass
    def expected_shortfall(self, ):
        pass
    def info_ratio(self, ):
        pass

    # def calculate_expected_shortfall_2(self, input_freq: str, x_period: int, alpha: float):
    #     if input_freq == 'Daily':
    #         frequency = 252
    #     elif input_freq == 'Monthly':
    #         frequency = 12

    #     returns = self.metrics['annualized_return']
    #     vol = self.metrics['annualized_volatility']
    #     mu_x = returns * np.sqrt(x_period / frequency)
    #     sigma_x = vol * np.sqrt(x_period / frequency) 

    #     ES = alpha ** -1 * norm.pdf(norm.ppf(alpha)) * sigma_x - mu_x

    #     return ES
