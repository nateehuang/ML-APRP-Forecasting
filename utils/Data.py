import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
from typing import List
import warnings

# Data Classes:
class DataSeries(object):
    ''' 
    A class used to represent one single time series.

    Attributes
    ----------
    ticker: str
        name of the time series

    category: str
        ETF/Stock/etc
        
    ts: pd.DataFrame
        index: Datetime
        values: prices/values
    
    Methods
    ----------
    '''
    # TODO: add more frequency
    freq_map = {'daily': 252,
                'monthly': 12,
                'yearly': 1}

    def __init__(self, category: str, freq: str, time_series: pd.DataFrame):
        '''
        Inits a DataSeries instant.
        
        Args
        ----------
        time_series: pd.DataFrame
            a single series of data including ticker, dates and prices

        '''
        if isinstance(category, str):
            self.category = category
        else:
            raise TypeError("Category must be a string")

        if freq in self.freq_map:
            self.freq = freq
        else:
            raise ValueError("Frequency: " + freq + " is not defined")

        if isinstance(time_series, pd.DataFrame) and time_series.shape[1] == 1:
            self.ts = time_series
        else:
            raise TypeError("time series must be pandas.DataFrame with dimension (x, 1)")
    
    def __len__(self):
        '''
        Obtain length of the time series.
        
        Returns
        ----------
        length: int
            The number of the prices in the series as a float. 
            
        '''
        return self.ts.shape[0]

    def __str__(self):
        '''
        frequency + ticker
        '''
        return str(self.freq + " " + self.get_ticker())

    def __add__(self, other):
        '''
        Return a DataSeries with both of ts in two DataSeries
        The two DataSerie must have the same category, same ticker and no duplicated date index

        Returns
        ----------
        res: DataSeries

        '''
        if not isinstance(other, DataSeries):
            raise TypeError("Can only add DataSeries object")
        if self.get_category() != other.get_category():
            raise ValueError("Two DataSeries category do not agree")
        if self.get_ticker() != other.get_ticker():
            raise ValueError("Two DataSeries' ticker do not agree")
        df1, df2 = self.get_ts(), other.get_ts()
        try:
            ts = df1.append(df2, verify_integrity = True)
            ts.sort_index(inplace=True)
        except ValueError:
            raise ValueError("Two DataSeries have overlap dates")
        return DataSeries(self.category, self.freq, ts)
        
    def __sub__(self, other):
        '''
        
        '''
        if not isinstance(other, DataSeries):
            raise TypeError("Can only subtract DataSeries object")
        if self.get_category() != other.get_category():
            raise ValueError("Two DataSeries category do not agree")
        if self.get_ticker() != other.get_ticker():
            raise ValueError("Two DataSeries' ticker do not agree")
        df1, df2 = self.get_ts(), other.get_ts()
        ts = df1[~ df1.index.isin(df2.index)]
        # check if all elements on df2 are subtracted
        if ts.shape[0] != df1.shape[0] - df2.shape[0]:
            raise AssertionError("Not all DataSeries are subtracted")
        return DataSeries(self.category, self.freq, ts)

    def get_ticker(self) -> str:
        '''
        Return the ticker of the underlying DataSeries 
        '''
        return self.ts.columns[0]

    def get_category(self) -> str:
        return self.category

    def get_freq(self) -> str:
        return self.freq

    def freq_to_int(self) -> int:
        return self.freq_map[self.freq]

    def get_ts(self):
        ''' 
        Return the dataframe of the series
        '''
        return self.ts.copy()
    
    def to_list(self):
        '''
        return a list of ts
        '''
        return self.get_ts().iloc[:, 0].values.tolist()

    def get_last_date(self):
        '''
        get the last index
        '''
        return self.ts.index[-1]
    
    def get_first_date(self):
        return self.ts.index[0]

    def mean(self) -> float:
        return self.ts.mean()[0]

    def get_return(self):
        '''
        Returns
        ----------
        pd.DataFrame
        '''
        return self.ts.pct_change().dropna()
    
    def price_to_return(self,):
        '''
        Returns
        ----------
        DataSeries
        '''
        ts = self.get_return()
        return DataSeries(self.category, self.freq, ts)
    
    def get_min(self):
        '''
        get the minimum item in the series
        '''
        return self.ts.min().values[0]

    def copy(self):
        '''
        Return a deep copy of the object

        Returns
        ----------
        copy: DataSeries
        '''
        return deepcopy(self)

    def to_Series(self) -> pd.Series:
        '''
        Return a pd.Series version of the data
        '''
        return self.ts.iloc[:, 0]

    def split(self, pct:float = None, numTest: int = None):
        '''
        Split the data as training and test sets

        Parameters
        ----------
        pct: float
            pecentage of data as training set, from 0 to 1

        Returns
        ---------
        train, test: (DataSeries, DataSeries)
        '''
        if pct and numTest:
            raise ValueError('Only one of the parameter can be passed')
        elif pct:
            end_ind = int(len(self) * pct)
        elif numTest:
            end_ind = len(self) - numTest
        else:
            raise ValueError('At least one of the parameters has to be passed')
        
        train = DataSeries(self.get_category(), self.freq, self.ts.iloc[:end_ind])
        test = DataSeries(self.get_category(), self.freq, self.ts.iloc[end_ind:])
        return train, test

    def trim(self, start_date: str, end_date: str):
        '''
        trim the DataSeries

        Returns
        ----------
        DataSeries
        '''
        return DataSeries(self.get_category(), self.freq, self.ts.loc[start_date: end_date])


class DataCollection(object):
    ''' 
    A class of a subset of time series.

    Attributes
    ----------
    label: str
        a description of this collection

    frequncy: str
        frequncy of the data
        daily
        monthly
        quarterly
        yearly
    
    collection: List[DataSeries]
        A list of DataSeries

    '''

    def __init__(self, 
                label: str,
                time_series_group: List[DataSeries]):
        '''
        Inits SeriesDataset Class.
        
        Args
        ----------
        label: str
            a description of this collection
        
        time_series_group:
            a collection of data series 

        path: str
            path to csv file
            csv format: datetime as row, ticker as column
        '''
        if isinstance(label, str):
            self.label = label
        else:
            raise TypeError("Label has to be a string")

        if len(time_series_group) == 0:
            raise ValueError("There is no item in the collection")

        # use dict to store DataSeries
        self.collection = {}
        # maintain a list of tickers
        self.tickers = []
        # use dict to store frequencies if there are multiple frequencies
        self.freq = {}
        # store the length of the collection
        self.length = 0
        # deep copy the DataSeries obejct inside the List
        for ts in time_series_group:
            if isinstance(ts, DataSeries):
                ticker = ts.get_ticker()
                self.collection[ticker] = ts.copy()
                self.tickers.append(ticker)
                self.freq[ticker] = ts.get_freq()
                self.length += 1
            else:
                raise TypeError("Must be DataSeries object in time_series_group")
        first = next(iter(self.freq.values()))
        if all(v == first for _, v in self.freq.items()):
            self.freq = first
        self.tickers.sort()

    def __len__(self):
        return self.length

    def __iter__(self):
        self._iter = -1
        return self

    def __next__(self):
        self._iter += 1
        if self._iter < len(self):
            return self[self._iter]
        else:
            raise StopIteration
    
    def __getitem__(self, key):
        return self.collection[self.tickers[key]]

    def __str__(self):
        return self.label
    
    def __add__(self, other):
        if not isinstance(other, DataCollection):
            raise ValueError("Can only add DataCollection object")
        if self.label != other.label:
            raise ValueError("Can only add DataCollection objects with the same label") 

        series_list = []
        for s2 in other:
            ticker = s2.get_ticker()
            s1 = self.get_series(ticker) 
            series = s1 + s2
            series_list.append(series)
        return DataCollection(self.label, series_list) 

    def __sub__(self, other):
        if not isinstance(other, DataCollection):
            raise ValueError("Can only subtract DataCollection object")
        if self.label != other.label:
            raise ValueError("Can only subtract DataCollection objects with the same label") 

        series_list = []
        for s2 in other:
            ticker = s2.get_ticker()
            s1 = self.get_series(ticker) 
            series = s1 - s2
            series_list.append(series)
        return DataCollection(self.label, series_list) 

    def select_tickers(self, tickers):
        lst_ticker = [self.get_series(t) for t in tickers]
        return DataCollection(self.label, lst_ticker)

    def get_freq(self):
        '''
        Return all the series freqs in this collection if various, 
        return single freq if all the same
        '''
        return self.freq

    def check_one_freq(self):
        return isinstance(self.freq, str)

    def get_series(self, ticker: str):
        '''
        Return a DataSeriese given a ticker

        Parameters
        ----------
        ticker: str

        Returns
        ----------
        DataSeries
        '''
        if ticker not in self.tickers:
            raise KeyError(ticker + " does not exist in this collection")
        return self.collection[ticker]

    def summary(self) -> pd.DataFrame:
        '''
        Get a summary of the collection: Length of each Time Series
        '''
        # TODO: to be implemented
        zipper = [(serie.get_ticker(), len(serie)) for serie in self]
        # unzip
        ticker, length = zip(*zipper)
        df = pd.DataFrame(list(length), index = ticker, columns = ['Length'])
        return df

    def split(self, pct: float = None, numTest: int = None):
        ''' 
        Cross validation preparation.
        
        split the data into training and test set. The function split the 
        whole data into three components: the training set, the validation 
        set, and the test set. We will construct the model on the training 
        set and validation set only. The test set is for performance 
        evaluation and will be used once.

        Args
        ----------
        pct: float
            pecentage of data as training set
        
        Returns
        ----------
        (training, test): (DataCollection, DataCollection)
        '''
        sp = [s.split(pct = pct, numTest = numTest) for s in self]
        train_zip, test_zip = zip(*sp)
        train = DataCollection(self.label, list(train_zip))
        test = DataCollection(self.label, list(test_zip))
        return train, test

    def trim(self, start_date:str, end_date:str):
        sp = [series.trim(start_date, end_date) for series in self]
        return DataCollection(self.label, sp)

    def price_to_return(self):
        '''
        Returns
        ----------
        DataCollection
        '''
        group = [series.price_to_return() for series in self]
        return DataCollection(self.label + 'return', group)
    
    def get_min(self):
        '''
        get a list of minimum item from the series in the collection

        Returns
        ----------
        List(float)
        '''
        return [series.get_min() for series in self]

    def ticker_list(self):
        '''
        Return a list of tickers

        Returns
        -----------
        self.tickers: List[str]
        '''
        return self.tickers

    def category_list(self):
        return [serie.get_category() for serie in self]

    def last_date_list(self):
        return [series.get_last_date() for series in self]

    def first_date_list(self):
        return [series.get_first_date() for series in self]
    
    def to_list(self):
        '''
        don't use yet
        '''
        return [series.to_list() for series in self]

    def add(self, data: DataSeries,):
        '''
        add a dataSereis
        '''
        if isinstance(data, DataSeries):
            tick = data.get_ticker()
            if tick not in self.tickers:
                self.collection[tick] = data.copy()
            else:
                warnings.warn(tick + " already exists in this collection. DataSeries is not added. Consider using '+' operator to add DataSeries")
        else:
            raise TypeError('Input should be an instance of DataSeries')

    def to_df(self) -> pd.DataFrame:
        '''
        Convert a DataCollection to pandas.DataFrame

        Parameters
        ----------
        dc: DataCollection


        Returns
        ----------
        DataFrame
        '''
        return pd.concat([serie.get_ts() for serie in self], axis = 1)

    def get_returns(self):
        ''' 
        
        Returns
        ----------
            pd.Dataframe of returns 
        '''
        return self.to_df().pct_change().dropna()

    def copy(self):
        '''
        return a deep copy of the object

        Returns
        ----------
        DataCollection
        '''
        return deepcopy(self)


