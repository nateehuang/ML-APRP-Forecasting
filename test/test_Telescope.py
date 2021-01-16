import unittest
import pandas as pd
import os
import utils.DataPreprocessing as preprocess
from utils.Data import DataSeries, DataCollection
from Models.Telescope import ModelTelescope

class Test_Telescope(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # An example of how to use Telescope model
        path_monthly = os.path.join('test','Data','Monthly')
        dic_monthly = preprocess.read_file(path_monthly)
        series_list = []
        for k, v in dic_monthly.items():
            df, cat = v
            df = preprocess.single_price(df, k)
            series_list.append(DataSeries(cat, 'monthly', df))
        self.collect = DataCollection('test1', series_list)

    def test_Telescope_forecast(self):
        # An example of how to use Telescope model
        collect = self.collect 
        m = ModelTelescope()
        train_dc, _ = collect.split(numTest = 12)
        m.train(train_dc)
        y_test = m.predict(12)
        y_test.to_df().to_csv('telescope_test_1.csv')
        
    def test_Telescope_recommend(self):
        collect = self.collect
        m = ModelTelescope()
        m.train_recommender(collect)
        print(m.rec_model)
    
    def test_Telescope_recommend_forecast(self):
        collect = self.collect
        m = ModelTelescope()
        train_dc, _ = collect.split(numTest=12)
        train_rec_dc, fit_dc = train_dc.split(pct=.5)
        m.train_recommender(train_rec_dc)
        m.train(fit_dc)
        y_test = m.predict(12)
        y_test.to_df().to_csv('telescope_test_2.csv')

if __name__ == "__main__":
    unittest.main()