import pandas as pd 
import numpy as np 
import utils.Optimization as Opt
from utils.Data import DataSeries, DataCollection

# Base Portfolio Objects
class Portfolio(object):
    ''' 
    Specialized and benchmark portfolio objects.
    
    Attributes
    ----------
    label: str
    solution: str
    
    optimizer: function from optimization.py
    tickers: List[str]
    intial_weights: pd.DataFrame
    weights: pd.DataFrame
        
    Methods
    ----------
    calculate_initial_weigths()
    rebalacing()
    
    '''

    # setup & utilities
    def __init__(self, label: str, solution: str):
        _portfolios_dict = {Opt.black_litterman_portfolio:{'black_litterman_portfolio'} ,
                            Opt.equal_portfolio: {'equal_portfolio'},
                            Opt.portfolio_opt: {'max_sharpe',
                                                'min_volatility',
                                                'max_quadratic_utility',
                                                'efficient_risk',
                                                'efficient_return'}
                            }
        # name of the portfolio
        self.label = label
        self.solution = solution
        # find optimizer 
        for key in _portfolios_dict:
            if self.solution in _portfolios_dict[key]:
                self.optimizer = key
                break
        
        self.tickers = None
        # weight when initiate portfolio
        self.initial_weight = None
        # store input_dc freq, avaliable after initial weights calculation
        self.input_freq = None
        # YET TO Implement:
        # a collection of rebalance weigths
        self.weights = [] 
        
    def get_label(self):
        return self.label
    
    def get_tickers(self):
        return self.tickers
    
    def get_freq(self):
        return self.input_freq

    def get_solution(self):
        return self.solution
    
    def get_initial_weight(self):
        return self.initial_weight

    def get_weights(self):
        return self.weights

    def get_optimizer(self):
        return self.optimizer

    # Optimization
    def calculate_initial_weight(self, input_dc: DataCollection, weight_bounds = (0,1), risk_aversion = 1, 
                                market_neutral = False, risk_free_rate = 0.0, target_volatility = 0.01, 
                                target_return = 0.11, returns_data = True, compounding = False):
        if not isinstance(input_dc.get_freq(), str):
            raise Exception("Optimization failed due to inconsistent series frequencies within input_dc.")
        else: 
            self.input_freq = input_dc.get_freq()
        if self.initial_weight is None:
            self.tickers = input_dc.ticker_list()
            self.initial_weight = self.optimizer(input_dc.to_df().dropna(), self.input_freq, self.solution,
                                                weight_bounds,risk_aversion, market_neutral, risk_free_rate, 
                                                target_volatility, target_return, returns_data, compounding)
        else:
            raise Exception("initial weight was already calculated")

    def rebalancing(self, new_input_dc: DataCollection, windows: int = 0):
        # df.rolling(windows).apply(Lambda:)
        pass
        
# classes:
# TO DO: BlackLitterman Portfolio Implement
class BlackLittermanPort(Portfolio):
    
    def __init__(self, label: str):
        super().__init__(label, 'black_litterman_portfolio')
    
    #override
    def calculate_initial_weight(self, input_dc):
        self.initial_weight = self.optimizer(input_dc.to_df().dropna(), risk_free_rate = 0.0, frequency=252)
   
class EqualPort(Portfolio):
    def __init__(self, label: str):
        super().__init__(label, 'equal_portfolio')
    
    #override
    def calculate_initial_weight(self, input_dc: DataCollection):
        if not isinstance(input_dc.get_freq(), str):
            raise Exception("Optimization failed due to inconsistent series frequencies within input_dc.")
        else: 
            self.input_freq = input_dc.get_freq()
        if self.initial_weight is None:
            self.tickers = input_dc.ticker_list()
            self.initial_weight = self.optimizer(input_dc.to_df().dropna())
        else:
            raise Exception("initial weight was already calculated")
    
class MaxSharpePort(Portfolio):
    def __init__(self, label):
        super().__init__(label,'max_sharpe')

class MinVolPort(Portfolio):
    def __init__(self, label):
        super().__init__(label,'min_volatility')

class InverseVarPort(Portfolio):
    def __init__(self, label):
        super().__init__(label,'inverse_variance')

class MaxQuaUtiPort(Portfolio):
    def __init__(self,  label):
        super().__init__(label,'max_quadratic_utility')
 
class EffRiskPort(Portfolio):
    def __init__(self, label):
        super().__init__(label,'efficient_risk')

class EffReturnPort(Portfolio):
    def __init__(self, label):
        super().__init__(label, 'efficient_return')
    