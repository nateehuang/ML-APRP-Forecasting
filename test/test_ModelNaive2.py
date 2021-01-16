import unittest
import pandas as pd
import utils.DataPreprocessing as preprocess
from utils.Data import DataSeries, DataCollection
from Models.ModelESRNN import ModelESRNN
from Models.ModelNaive2 import ModelNaive2
import os

class Test_Naive2(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """ a = pd.DataFrame([10.2, 12, 32.1, 9.32], columns=['ABC'], 
                            index=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-04-01'])) 
        a.index.name = 'Date'
        self.a_series = DataSeries('ETF', 'monthly', a)

        b = pd.DataFrame([2.3, 3.6, 4.5], columns=['KKK'],
                            index=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01',]))
        b.index.name = 'Date'
        self.b_series = DataSeries('Bond', 'monthly', b)
        self.collect = DataCollection('trial', [self.a_series, self.b_series])
 """
    def test_Naive2(self):
        path_monthly = os.path.join('test','Data','Monthly')
        dic_monthly = preprocess.read_file(path_monthly)
        series_list = []
        for k, v in dic_monthly.items():
            df, cat = v
            df = preprocess.single_price(df, k)
            series_list.append(DataSeries(cat, 'monthly', df))
        collect = DataCollection('test1', series_list)
        train_dc, test_dc = collect.split(numTest = 12)
        m = ModelNaive2(12, train_dc, test_dc)
        y_hat_Naive2_dc = m.fit_and_generate_prediction(12, freq='MS')

        y_hat_Naive2_dc.to_df().to_csv('test_Naive2_result.csv')


if __name__ == "__main__":
    unittest.main()