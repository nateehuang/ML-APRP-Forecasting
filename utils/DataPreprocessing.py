import pandas as pd
from os import listdir
import os, sys
from datetime import datetime

# Read raw data:
def read_file(file_location):
    ''' Read time series data from files.

    The function should be appliable for all types of data file including but
    not limited to .cvs/.xls/.txt/etc. The output list may include different
    types of data such as string and float.

    Args
    ----------
        file_location: A path to where the data is stored.
    
    Returns
    ----------
        A Python dictionary of dataframes and corresponding data category.
    
    '''
    df_dic = {}
    count = 0
    recover_list = []
    ticker_list = []
    
    for root, _, files in os.walk(file_location):
        for name in files:
            if name == '.DS_Store':
                continue
            file_path = os.path.join(root,name)
            # make the path consistent among Window and Mac system
            string_path = file_path.replace('/', '\\')
            # read category
            category = str(string_path).split('\\')[-2]

            df = read_single_file(name, file_path)
            df = single_price(df,name[:-4])
            df, recover_min = simple_imputation(df)
            recover_list.append(recover_min)
            ticker_list.append(name[:-4])

            # new_name = name.replace(' ','_')
            df_dic[name[:-4]] = [df,category]
            count+=1
            #print('the '+ str(count)+'th file '+name+' is completed under category '+category)
    #print(str(count)+' files are read into a Dictionary of pd.DataFrame.')

    return df_dic, recover_list, ticker_list

def simple_imputation(df):
    df = df.pct_change().dropna()
    recover_min = df.min().item()
    df = df - recover_min + 1
    return df, recover_min
    # return df.dropna()
    
def csv_to_pd(file_path):
    dp = pd.to_datetime
    df = pd.read_csv(file_path, date_parser=dp, index_col=['Date']).astype(float)
    return df

def excel_to_pd(file_path):
    df = pd.read_excel(file_path, skiprows=6)[:-1]
    df = df[df['Effective date '].notna()].set_index('Effective date ').astype(float) # drop rows without date
    return df

def read_single_file(file_name, file_path):
    if file_name[-4:] == '.csv':
        df = csv_to_pd(file_path)
    else:
        df = excel_to_pd(file_path)
    return df
    
def single_price(df,ticker):
    ''' Make one ticker to correspond to only one line of prices.
    
    There are some dataframes that include Open/High/Low/Close/Adj Close/etc. 
    In time series analysis, we only need one line of prices for each ticker. 
    This function is to obtain that one line of prices from different merging
    rule. 

    What's more, vwap/twap are much more used on the trading and short 
    investment strategies side than on investment and wealth management 
    side (which requires a medium and long term for both investing and 
    for forecasting). In most cases we rely on closing daily prices when 
    available; for a few asset classes we may only have weekly, monthly, 
    or even quarterly.

    Args
    ----------
    df: pd.DataFrame
        a single pd.Dataframe with ochl etc columns or one with only one
        column

    Returns
    ----------
        new dataframes
    '''

    if len(df.columns) != 1:
        new_df = df[['Close']]
    else:
        new_df = df
    
    new_df.columns = [ticker]

    return new_df


def imputation(rule=None):
    ''' 
    Deal with NaN.

    Since DataCollection is constructed by combining different DataSeries
    objects with different lengths, it might contain NaN values in __init__.
    We need to consider how to deal with those NaN points so that the 
    DataCollection.prices_df dataframe can be used as an input of models
    properly. 

    Args
    ----------
    rule: str
        Default is 'None' will drop all the NaN values. 
        ‘fillzero'
        'fillforward'
        'fillbackward'
        'fillaverage'

    Returns
    ----------
        update self.ts_df without NaN points

    '''
    # if rule == None:
    #     df = self.prices_df.dropna()
    # elif rule == 'fillzero':
    #     df = self.prices_df.fillna()
    # elif rule == 'fillforward':
    #     df = self.prices_df.fillna(method='ffill')
    # elif rule == 'fillbackward':
    #     df = self.prices_df.fillna(method='bfill')
    # elif rule == 'fillaverage':
    #     df = self.prices_df.fillna(self.prices_df.mean())
    # else:
    #     df = self.prices_df
    #     print('Try other methods. Original df is kept.')
    # self.prices_df = df
    pass
