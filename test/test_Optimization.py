import unittest             # The test framework
from pandas.util.testing import assert_frame_equal
import utils.Optimization as Opt    # The code to test
from utils.Data import DataSeries, DataCollection
import utils.DataPreprocessing as DataPreprocessing
import pypfopt.exceptions as exceptions
import pandas as pd
import numpy as np
import os

class Test_Optimization(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        path_monthly = os.path.join('test','Data','Monthly') 
        dic_monthly = DataPreprocessing.read_file(path_monthly)

        n_assets = 4
        time_series_group = []
        
        for i in range(n_assets):
            df = dic_monthly[list(dic_monthly.keys())[i]]
            ds = DataSeries('ETF', 'monthly', df[0])
            time_series_group.append(ds)

        input_dc_test = DataCollection(label='Test Collection', time_series_group=time_series_group)
        self.input_dc = input_dc_test
        self.input_freq = input_dc_test.get_freq()
        self.input_df = self.input_dc.to_df().dropna()

        self.a = pd.DataFrame([10, 12, 32, 9, 11, 9], columns=['fakeSPY'],
            index=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-04-01','2020-05-01', '2020-06-01']))
        self.a_series = DataSeries('ETF', self.input_freq, self.a)
        self.b = pd.DataFrame([1, 1.2, 3.2, 0.9], columns=['fakeTreasury'],
                index=pd.to_datetime(['2019-12-01', '2020-02-01', '2020-03-01','2020-04-01']))
        self.b_series = DataSeries('Bond', self.input_freq, self.b)
        self.c_collection = DataCollection('trial', [self.a_series, self.b_series])
        self.c_df = self.c_collection.to_df().interpolate(method='linear', axis=0)
    
    def test_calculate_annualized_expected_returns(self):
        res = Opt.calculate_annualized_expected_returns(self.c_df, self.input_freq)
        expected = self.c_df.pct_change().mean()*12
        assert_frame_equal(res.to_frame(), expected.to_frame())
    
    def calculate_annualized_return_covariance(self):
        res = Opt.calculate_annualized_return_covariance(self.c_df, self.input_freq)
        expected = self.c_df.pct_change().cov()*12
        assert_frame_equal(res.to_frame(), expected.to_frame())  

    def test_equal_portfolio(self):
        wts = Opt.equal_portfolio(self.input_df)
        self.assertEqual(type(wts), pd.DataFrame)
        self.assertEqual(wts.shape, (1, 4))
        for w in wts.values[0]:
            self.assertTrue(w >= 0 and w <= 1)
        self.assertEqual(round(wts.sum(axis=1)[0]), 1)
        self.assertTrue(wts.eq(wts.iloc[:, 0], axis=0).all(1).item())

    def test_black_litterman_portfolio(self):
        pass

    def test_portfolio_general(self):
        
        methods_list = ['max_sharpe','min_volatility','max_quadratic_utility',
                        'efficient_risk','efficient_return']
        
        for method in methods_list:
            try:
                wts = Opt.portfolio_opt(self.input_df, self.input_freq, solution = method)
            except:
                continue
    
            self.assertEqual(type(wts),pd.DataFrame)
            self.assertEqual(wts.shape, (1, 4))
            for w in wts.values[0]:
                self.assertTrue(w >= 0 and w <= 1)
            self.assertEqual(round(wts.sum(axis=1)[0]), 1)
    
if __name__ == "__main__":
    unittest.main()