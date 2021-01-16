import unittest
import pandas as pd
import utils.DataPreprocessing as preprocess
from utils.Data import DataSeries, DataCollection
from Models.ModelESRNN import ModelESRNN
import os


class Test_ESRNN(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        a = pd.DataFrame([10.2, 12, 32.1, 9.32], columns=['ABC'], 
                            index=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-04-01'])) 
        a.index.name = 'Date'
        self.a_series = DataSeries('ETF', 'monthly', a)

        b = pd.DataFrame([2.3, 3.6, 4.5], columns=['KKK'],
                            index=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01',]))
        b.index.name = 'Date'
        self.b_series = DataSeries('Bond', 'monthly', b)
        self.collect = DataCollection('trial', [self.a_series, self.b_series])

    def test_data_reformat(self):

        X, y = ModelESRNN().data_reformat(self.collect)

        
        expect = pd.DataFrame({'unique_id':['ABC']*4 + ['KKK']*3,
                                'ds':X['ds'],
                                'x':['ETF']*4 + ['Bond']*3,
                                'y':[10.20, 12.00, 32.10, 9.32, 2.3, 3.6, 4.5]})

        expect_x = expect[['unique_id', 'ds', 'x']]
        expect_y = expect[['unique_id', 'ds', 'y']]
        assert(X.equals(expect_x))
        assert(y.equals(expect_y))

    @staticmethod
    def compareSeries(a, b):
        flag = True
        if not isinstance(a, DataSeries):
            print("\n The first item is not a DataSeries object")
            return False
        if not isinstance(b, DataSeries):
            print("\n The Second item is not a DataSeries object")
            return False
        if a == b:
            print("\n The two items are the same object")
            flag = False
        if len(a) != len(b):
            print("\n The two items does not have the same length")
            flag = False

        if str(a) != str(b):
            print("\n The two items does not have the same ticker")
            flag = False

        if a.get_category() != b.get_category():
            print("\n The two items does not have the same category")
            flag = False
        
        if not a.get_ts().equals(b.get_ts()):
            print("\n The two items does not have the same time series")
            flag = False

        if not a.get_freq() == b.get_freq():
            print("\n The two items does not have the same frequency")
            flag = False

        return flag

    def test_data_reverse_to_dc(self):
        # test reverse
        result = pd.DataFrame({'unique_id':['ABC'] * 4 + ['KKK'] * 3, 
                               'ds':pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-04-01',
                                                    '2020-01-01','2020-02-01','2020-03-01']),
                               'x': ['ETF'] * 4 + ['Bond'] * 3,
                               'y_hat': [10.2, 12, 32.1, 9.32,
                                     2.3, 3.6, 4.5]})
        dc = ModelESRNN().to_dc(result, 'trial', {'ABC': 'monthly', 'KKK': 'monthly'})
        for this, that in zip(dc, self.collect):
            assert(self.compareSeries(this, that))

    def test_ESRNN(self):
        # An example of how to use ESRNN
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        path_daily = os.path.join('test','Data','daily')
        dic_daily = preprocess.read_file(path_daily)
        series_list = []
        for k, v in dic_daily.items():
            df, cat = v
            df = preprocess.single_price(df, k)
            series_list.append(DataSeries(cat, 'daily', df))
        collect = DataCollection('test1', series_list)
        m = ModelESRNN(max_epochs = 5, seasonality = [], batch_size = 64, input_size = 12, output_size = 12, device = device)
        train_dc, test_dc = collect.split(numTest = 12)

        m.train(train_dc)
        y_test = m.predict(test_dc)
        assert(isinstance(y_test, DataCollection))
        y_test_df = y_test.to_df()
        y_test_df.to_csv('predict_result.csv')     

if __name__ == "__main__":
    unittest.main()