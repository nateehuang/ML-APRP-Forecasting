import unittest             # The test framework
import utils.Optimization as Opt    # The code to test
from utils.Data import DataSeries, DataCollection
import utils.DataPreprocessing as DataPreprocessing
import utils.Portfolio_Performance as PP
import utils.Portfolio as port
import pypfopt.exceptions as exceptions
import pandas as pd
import numpy as np
import os

class Test_Portfolio_Performance(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # fake data by ZZ Daily
        self.a_series = DataSeries('ETF', 'daily', pd.DataFrame([10.0, 15.0, 20.0, 30.0], columns=['ABC'], 
                            index=pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03','2020-01-04'])))
        self.b_series = DataSeries('Bond', 'daily', pd.DataFrame([1.0, 3.5, 4.5], columns=['KKK'],
                            index=pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03',])))
        self.collect = DataCollection('trial', [self.a_series, self.b_series])
        
        d = {'Initial weights': [0.6, 0.4]}
        self.weights = pd.DataFrame(data=d).T
        self.weights = self.weights.rename(columns={0: 'ABC', 1: 'KKK'})

        self.p = port.EqualPort("test equal port") 
        self.p.calculate_initial_weight(self.collect)

        # Monthly
        path_monthly = os.path.join('test','Data','Monthly') 
        dic_monthly = DataPreprocessing.read_file(path_monthly)

        n_assets = 4
        time_series_group = []
        for i in range(n_assets):
            df = dic_monthly[list(dic_monthly.keys())[i]]
            ds = DataSeries(df[1], 'monthly', df[0])
            time_series_group.append(ds)
        input_dc_test = DataCollection(label='Test Collection', time_series_group=time_series_group)
        self.input_dc = input_dc_test
        self.input_freq = input_dc_test.get_freq()
        self.input_df = self.input_dc.to_df()
        self.n_asset = len(self.input_df.columns)
        input_weights = [[1/self.n_asset] * self.n_asset]
        input_weights_df  = pd.DataFrame(input_weights, columns=self.input_df.columns,index=['Initial weights'])
        self.input_weights_df = input_weights_df

    def test_annualized_return(self):        
        ans = PP.annualized_return(self.weights, self.collect.to_df().dropna(), 'daily')

        output_return_df = self.collect.to_df().dropna().pct_change().dropna()
        annual_return = output_return_df.mean() * 252
        ans_new = annual_return @ self.weights.T
        
        self.assertAlmostEqual(ans_new[0], 203.4)
        self.assertAlmostEqual(ans, ans_new[0])

        res2 = PP.annualized_return(self.input_weights_df, self.input_df, self.input_freq)
        expected2 = np.dot(self.input_weights_df, self.input_df.pct_change().mean()*12).item()
        self.assertEqual(res2, expected2)
    
    def test_annualized_volatility(self):
        ans = PP.annualized_volatility(self.weights, self.collect.to_df().dropna(), 'daily')

        cov = self.collect.to_df().dropna().pct_change().dropna().cov() * 252
        ans_new = (self.weights @ cov @ self.weights.T).iloc[0][0] ** 0.5   
            
        self.assertAlmostEqual(ans, ans_new)

        res2 = PP.annualized_volatility(self.input_weights_df, self.input_df, self.input_freq)
        expected2 =  np.sqrt(np.dot(self.input_weights_df, np.dot(self.input_df.pct_change().cov()*12, self.input_weights_df.T)).item())
        self.assertAlmostEqual(res2, expected2)

    def test_sharpe_ratio(self):
        ans = PP.sharpe_ratio(0.6, 0.2, 0.03)
        ans_new = (0.6-0.03)/0.2
        self.assertAlmostEqual(ans, ans_new)

    def test_PnL(self):
        ans = PP.PnL(self.weights, self.collect.to_df().dropna())

        output_return_df = self.collect.to_df().dropna().pct_change().dropna()
        ans_1 = output_return_df.iloc[0][0] * self.weights.iloc[0][0] + output_return_df.iloc[0][1] * self.weights.iloc[0][1]
        ans_2 = output_return_df.iloc[1][0] * self.weights.iloc[0][0] + output_return_df.iloc[1][1] * self.weights.iloc[0][1]

        self.assertAlmostEqual(ans.iloc[0][0], ans_1)
        self.assertAlmostEqual(ans.iloc[1][0], ans_2)

    def test_max_drawdown(self):
        price = {'PnL': [75, 33, 35, 25, 80, 100, 95, 78, 72, 62, 65, 60, 42, 50]}
        pnl = pd.DataFrame(data=price).pct_change().dropna()

        ans = PP.max_drawdown(pnl)
        ans_new = (25-75)/75

        self.assertAlmostEqual(ans, ans_new)

    def test_partial_moment(self):
        pnl = PP.PnL(self.weights, self.collect.to_df().dropna())
        pm = PP.partial_moment(pnl, threshold=0.6)

        length = pnl.shape[0]
        threshold = 0.6
        diff_df = threshold - pnl
        drop_minus = diff_df[diff_df>=0].dropna()
        pm_new = ((drop_minus ** 2).sum() / length).item()

        self.assertAlmostEqual(pm, 0.0408163265)
        self.assertAlmostEqual(pm, pm_new)

    def test_PP_class(self):
        self.assertEqual(self.p.get_freq(), self.collect.get_freq())

        pp = PP.PortfolioPerformance(self.p, self.collect)
        pp.annualized_return()
        pp.annualized_volatility()
        pp.annualized_sharpe_ratio()
        # pp.print_metrics()
        # pp.get_metrics('annualized_return')
        pp.PnL()
        # print(pp.metrics['PnL'])
        pp.max_drawdown() # 0 since test data always increasing, but function tested

        self.assertEqual(pp.get_metrics("annualized_return"), 228)
        self.assertEqual(pp.get_metrics("PnL").iloc[0][0], 1.5)
        pp.print_metrics()
        pp.get_metrics('PnL')

        self.assertEqual(pp.metrics['annualized_return'], 228)
        self.assertAlmostEqual(pp.metrics['annualized_volatility'], 13.3630621)
        self.assertAlmostEqual(pp.metrics['sharpe_ratio'], 228/13.3630621)
        self.assertAlmostEqual(pp.metrics['PnL'].iloc[0][0], 1.5)
        self.assertAlmostEqual(pp.metrics['PnL'].iloc[1][0], 0.30952380952380953)
        self.assertEqual(pp.metrics['max_drawdown'], 0)

        # sortino
        pp.sortino_ratio(threshold=0.6)
        
        d = {'Initial weights': [0.5, 0.5]}
        self.weights2 = pd.DataFrame(data=d).T
        self.weights2 = self.weights2.rename(columns={0: 'ABC', 1: 'KKK'})
        pnl = PP.PnL(self.weights2, self.collect.to_df().dropna())

        threshold = 0.6
        expected = pp.metrics['annualized_return']
        lpm_sortino = PP.partial_moment(pnl, threshold, order = 2, lower = True) ** 0.5

        ans_sortino = (expected - threshold) / lpm_sortino

        self.assertAlmostEqual(pp.metrics['sortino_ratio'], ans_sortino)

        # omega
        pp.omega_ratio(threshold=0.6)
        lpm_omega = PP.partial_moment(pnl, threshold, order = 1, lower = True)
        ans_omega = ((expected - threshold) / lpm_omega) + 1

        self.assertAlmostEqual(pp.metrics['omega_ratio'], ans_omega)

# TODO: test max_sharpe gives the maximum sharpe ratio among all implemented portfolio


if __name__ == "__main__":
    unittest.main()
