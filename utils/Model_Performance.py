from Models.Model import BaseModel
from utils.Data import DataCollection
from utils.Portfolio import Portfolio
from typing import List
import pandas as pd
import numpy as np

def check_input_df_tickers(dfs: List[pd.DataFrame]) -> bool: 
    '''
    Check tickers in dfs

    Returns
    ----------
    True if all dfs have same tickers
    False o/w
    '''
    tickers = list(dfs[0].columns)
    return all(list(df.columns) == tickers for df in dfs if df.values.all())

def MASE(y_df: pd.DataFrame, y_hat_df: pd.DataFrame, 
         y_insample_df: pd.DataFrame, seasonality = 1) -> float:
    '''
    Mean Absolute Scaled Error
    Naïve 1:
        A random walk model, assuming that future values will be the same
        as that of the last known observation

    Args
    ----------
    y_df: pd.DataFrame
        actual test data OOS
    y_hat_df: pd.DataFrame
        predicted values OOS
    y_insample_df: pd.DataFrame
        actual training data in-sample
    seasonality=1: int
        for naive predicting
        if monthly data, use 12
        if daily/weekly/yearly, use 1
        if quarterly, use 4

    Returns
    ----------
    MASE: float

    '''
    forecast_error_df = np.abs(y_df - y_hat_df)
    naive_error_df = y_insample_df.diff(periods = seasonality).abs()
    up = forecast_error_df.stack().mean()
    low = naive_error_df.stack().mean()
    MASE = up/low
    return MASE

def RMSE(y_df: pd.DataFrame, y_hat_df: pd.DataFrame) -> float:
    '''Root Mean Square Error
    Args
    ----------
    y_df: pd.DataFrame
        actual test data
    y_hat_df: pd.DataFrame
        predicted values
    
    Returns
    -----------
    RMSE: float

    '''
    squared_error_df = (y_df - y_hat_df)**2
    mean = squared_error_df.stack().mean()
    RMSE = np.sqrt(mean)
    return RMSE
    
def MAPE(y_df: pd.DataFrame, y_hat_df: pd.DataFrame) -> float:
    '''
    Mean Absolute Percentage error

    Args
    ----------
    y_df: pd.DataFrame
        actual test data
    y_hat_df: pd.DataFrame
        predicted values
    
    Returns
    -----------
    MAPE: float
    '''
    pct_df =  np.abs(y_df - y_hat_df)/ np.abs(y_df)
    MASP = pct_df.stack().mean()
    return MASP

def sMAPE(y_df: pd.DataFrame, y_hat_df: pd.DataFrame) -> float:
    '''
    symmetric Mean Absolute Percentage Error

    Args
    ----------
    y_df: pd.DataFrame
        actual test data OOS
    y_hat_df: pd.DataFrame
        predicted values OOS

    Returns
    ----------
    sMAPE: float

    '''
    up = np.abs(y_df - y_hat_df)
    low = np.abs(y_df) + np.abs(y_hat_df)
    pct_df = up/low
    mean = pct_df.stack().mean()
    sMAPE = mean
    return sMAPE

def OWA(y_df: pd.DataFrame, y_hat_df: pd.DataFrame, 
        y_insample_df: pd.DataFrame, y_naive2_hat_df: pd.DataFrame, seasonality = 1) -> float:
    '''
    Mean Absolute Percentage error
    Naïve 2:
        Like Naïve 1 but the data are seasonally adjusted, if needed, 
        by applying a classical multiplicative decomposition. A 90% 
        autocorrelation test is performed to decide whether the data 
        are seasonal.

    Args
    ----------
    y_df: pd.DataFrame
        actual test data OOS
    y_hat_df: pd.DataFrame
        predicted values OOS
    y_insample_df: pd.DataFrame
        actual training data in-sample
    y_naive2_hat_df: pd.DataFrame
        predicted value from naive2 forecast approach OOS
    seasonality: int
    
    Returns
    -----------
    OWA: float
    '''
    model_mase = MASE(y_df, y_hat_df, y_insample_df, seasonality)
    model_smape = sMAPE(y_df, y_hat_df)
    naive2_mase = MASE(y_df, y_naive2_hat_df, y_insample_df, seasonality)
    naive2_smape = sMAPE(y_df, y_naive2_hat_df) 
    model_owa = ((model_mase/naive2_mase) + (model_smape/naive2_smape))/2
    return model_owa

def R2(y_df: pd.DataFrame, y_hat_df: pd.DataFrame) -> float:
    '''
    R-squared OOS

    Args
    ----------
    y_df: pd.DataFrame
        actual test data OOS
    y_hat_df: pd.DataFrame
        predicted values OOS
    
    Returns
    ----------
    R2: float
    '''
    residuals = y_df - y_hat_df
    SSR = (residuals**2).stack().sum()
    SST = y_df.stack().var()*(len(y_df.stack())-1)
    R2 = 1 - SSR/SST
    return R2

def Theil_U2(y_df: pd.DataFrame, y_hat_df: pd.DataFrame) -> float:
    '''
    Theil_U2 Forecast Coefficient

    Args
    ----------
    y_df: pd.DataFrame
        actual test data OOS
    y_hat_df: pd.DataFrame
        predicted values OOS

    Returns
    ----------
    Theil_U2: float

    '''
    forecast_error_df = y_hat_df - y_df
    y_df_shift = y_df.shift(1)
    up_df = (forecast_error_df/y_df_shift)**2
    up = np.sqrt(up_df.stack().sum()).item()

    low_error_df = y_df.diff(1)
    low_df = (low_error_df/y_df_shift)**2
    low = np.sqrt(low_df.stack().sum()).item()
    
    U2 = up/low
    return U2

def Theil_U1(y_df: pd.DataFrame, y_hat_df: pd.DataFrame) -> float:
    '''
    Theil_U1 Forecast Coefficient

    Args
    ----------
    y_df: pd.DataFrame
        actual test data OOS
    y_hat_df: pd.DataFrame
        predicted values OOS

    Returns
    ----------
    Theil_U2: float

    '''
    error2_df = (y_hat_df - y_df)**2
    up = np.sqrt(error2_df.stack().mean())

    low_1 = np.sqrt((y_df**2).stack().mean())
    low_2 = np.sqrt((y_hat_df**2).stack().mean())
    
    U1 = up/(low_1+low_2)
    return U1

class ModelPerformance(object):
    ''' Functions to generate our metric for model evaluation.
    
    Attributes
    ----------
   
    model: Model obj
    metrics: dictionary
    
    Methods
    ----------
     
    '''
    def __init__(self, label: str, 
                       seasonality: int, 
                       y_dc: DataCollection, 
                       y_hat_dc: DataCollection, 
                       y_insample_dc: DataCollection,
                       y_naive2_hat_dc = None):
        '''
        Args
        ----------
        label: str
            description of the target model
        y_df: pd.DataFrame
            actual test data OOS: test_df
        y_hat_df: pd.DataFrame
            predicted values OOS: forecast_df
        y_insample_df: pd.DataFrame
            actual training data in-sample: train_df
        y_naive2_hat_df: pd.DataFrame
            predicted value from native2 forecast approach OOS: naive2_df

        '''
        self.label = label
        self.seasonality = seasonality
        self.metrics = {}
        
        self.y_df = y_dc.to_df() # testing data
        self.y_hat_df = y_hat_dc.to_df() # forecasts
        self.y_insample_df = y_insample_dc.to_df() # training data
        # naive2 forecast : if we decide not to use OWA, ignore 
        if y_naive2_hat_dc is not None: 
            self.y_naive2_hat_df = y_naive2_hat_dc.to_df()
        else: 
            self.y_naive2_hat_df = pd.DataFrame([False], index = ['Naive2 None'])
        # check tickers
        if not check_input_df_tickers([self.y_df, self.y_hat_df, self.y_insample_df, self.y_naive2_hat_df]):
            raise ValueError('Tickers in input dfs do not match!')
        
    def MASE(self):
        self.metrics['MASE'] = MASE(self.y_df, self.y_hat_df, self.y_insample_df, seasonality = self.seasonality)
        
    def RMSE(self):
        self.metrics['RMSE'] = RMSE(self.y_df, self.y_hat_df)
       
    def MAPE(self):
        self.metrics['MAPE'] = MAPE(self.y_df, self.y_hat_df)
        
    def sMAPE(self):
        self.metrics['sMAPE'] = sMAPE(self.y_df, self.y_hat_df)
        
    def OWA(self):
        if self.y_naive2_hat_df.values[0][0] == False:
            self.metrics['OWA'] = None
        else:
            self.metrics['OWA']= OWA(self.y_df, self.y_hat_df, self.y_insample_df, self.y_naive2_hat_df, seasonality = self.seasonality)
        
    def R2(self):
        self.metrics['R2'] = R2(self.y_df, self.y_hat_df)
    
    def Theil_U1(self):
        self.metrics['Theil_U1'] = Theil_U1(self.y_df, self.y_hat_df)
    def Theil_U2(self):
        self.metrics['Theil_U2'] = Theil_U2(self.y_df, self.y_hat_df)

    def generate_all_metrics(self):
        self.MASE()
        self.RMSE()
        self.MAPE()
        self.sMAPE()
        self.OWA()
        self.R2()
        self.Theil_U2()
        self.Theil_U1()
        return self.metrics