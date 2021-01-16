import unittest             # The test framework
from utils.Portfolio import Portfolio, MaxSharpePort, EqualPort, BlackLittermanPort  # The code to test
import utils.Optimization as Opt
from utils.Data import DataSeries, DataCollection
import pandas as pd
import numpy as np

class Test_Portfolio(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # random some sample data
        np.random.seed(123)
        n_assets = 4

        time_series_group = []
        for i in range(n_assets):
            rows,cols = 1000,1
            data = np.random.rand(rows, cols) # use other random functions to generate values with constraints
            tidx = pd.date_range('2019-01-01', periods=rows, freq='MS') # freq='MS'set the frequency of date in months and start from day 1. You can use 'T' for minutes and so on
            ID = 'FakeStock_' + str(i+1)
            df = pd.DataFrame(data, columns=[ID], index=tidx)
            ds = DataSeries(category='Stock', freq = 'monthly', time_series=df)
            time_series_group.append(ds)
        input_dc_test = DataCollection(label='Test Collection', time_series_group=time_series_group)
        self.input_dc = input_dc_test

        # for exception test
        for i in range(2):
            rows,cols = 1000,1
            data = np.random.rand(rows, cols) # use other random functions to generate values with constraints
            tidx = pd.date_range('2019-01-01', periods=rows, freq='D') # freq='MS'set the frequency of date in months and start from day 1. You can use 'T' for minutes and so on
            ID = 'FakeStock_Daily_' + str(i+1)
            df = pd.DataFrame(data, columns=[ID], index=tidx)
            ds = DataSeries(category='Stock', freq = 'daily', time_series=df)
            time_series_group.append(ds)
        input_dc_test_2 = DataCollection(label='Test Collection 2', time_series_group=time_series_group)
        self.input_dc_2 = input_dc_test_2
    
    def test_single_dc_freq(self):
        assert(self.input_dc.get_freq()=='monthly')
        assert(self.input_dc_2.get_freq()=={'FakeStock_1':'monthly','FakeStock_2':'monthly',
                                            'FakeStock_3':'monthly','FakeStock_4':'monthly',
                                            'FakeStock_Daily_1':'daily','FakeStock_Daily_2':'daily'})

    def test_initiate_base(self):
        portfolio_1 = Portfolio(label='portfolio_1', solution= 'max_sharpe')
        assert(portfolio_1.get_solution()=='max_sharpe')
        assert(portfolio_1.get_initial_weight() == None)
        assert(portfolio_1.get_optimizer() == Opt.portfolio_opt)
        assert(portfolio_1.get_tickers() == None)
        assert(portfolio_1.get_freq() == None)
        assert(portfolio_1.get_weights() == [])
        assert(portfolio_1.get_label() == 'portfolio_1')
        
    def test_initiate_child(self):
        portfolio_1 = MaxSharpePort('label_1')
        portfolio_2 = EqualPort('label_2')
        # portfolio_3 = BlackLittermanPort(input_dc = self.input_dc)

        assert(portfolio_1.get_solution()=='max_sharpe')
        assert(portfolio_1.get_initial_weight() == None)
        assert(portfolio_1.get_optimizer() == Opt.portfolio_opt)
        assert(portfolio_1.get_tickers() == None)
        assert(portfolio_1.get_freq() == None)
        assert(portfolio_1.get_weights() == [])
        assert(portfolio_1.get_label() == 'label_1')

        assert(portfolio_2.get_solution()=='equal_portfolio')
        assert(portfolio_2.get_initial_weight() == None)
        assert(portfolio_2.get_optimizer() == Opt.equal_portfolio)
        assert(portfolio_2.get_tickers() == None)
        assert(portfolio_2.get_freq() == None)
        assert(portfolio_2.get_weights() == [])
        assert(portfolio_2.get_label() == 'label_2')

        '''
        assert(portfolio_3.get_solution()=='black_litterman_portfolio')
        assert(portfolio_3.get_initial_weight() == None)
        assert(portfolio_3.get_optimizer() == Opt.black_litterman_portfolio)
        '''
    
    def test_calculate_initial_weights(self):
        portfolio_1 = MaxSharpePort('label_1')
        assert(portfolio_1.get_initial_weight() == None)
        assert(portfolio_1.get_freq() == None)
        portfolio_1.calculate_initial_weight(self.input_dc)
        assert(type(portfolio_1.get_initial_weight()) == pd.DataFrame)
        assert(portfolio_1.get_initial_weight().shape[0]==1)
        assert(portfolio_1.get_tickers() == self.input_dc.ticker_list())
        assert(portfolio_1.get_freq() == 'monthly')

        portfolio_2 = EqualPort('label_2')
        assert(portfolio_2.get_initial_weight() == None)
        assert(portfolio_2.get_freq() == None)
        portfolio_2.calculate_initial_weight(self.input_dc)
        assert(type(portfolio_2.get_initial_weight()) == pd.DataFrame)
        assert(portfolio_2.get_initial_weight().shape[0]==1)
        assert(portfolio_2.get_tickers() == self.input_dc.ticker_list())
        assert(portfolio_2.get_freq() == 'monthly')
    
    def test_raise_error(self):
        portfolio_1 = MaxSharpePort('label_1')
        with self.assertRaises(Exception) as context:
            portfolio_1.calculate_initial_weight(self.input_dc_2)
        self.assertTrue("Optimization failed due to inconsistent series frequencies within input_dc." in str(context.exception))
        assert(portfolio_1.get_initial_weight() == None)
        assert(portfolio_1.get_freq()==None)

        portfolio_1.calculate_initial_weight(self.input_dc)
        with self.assertRaises(Exception) as context2:
            portfolio_1.calculate_initial_weight(self.input_dc)
        self.assertTrue("initial weight was already calculated" in str(context2.exception))
        

    
if __name__ == "__main__":
    unittest.main()



