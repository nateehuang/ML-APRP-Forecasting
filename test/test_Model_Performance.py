import unittest
import datetime
import pandas as pd 
import numpy as np
import os
import utils.DataPreprocessing as DP
import utils.Model_Performance as MP
import Models.ModelNaive2 as MN
from Models.ModelESRNN import ModelESRNN
from utils.Data import DataSeries, DataCollection

class Test_Model_Performance(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d1 = {'test': [75.0, 72, 69, 70, 76, 71, 74, 78, 72, 62]}
        d2 = {'test': [74, 70.5, 69, 71, 74.5, 74, 75, 71, 65, 64]}
        self.actual = pd.DataFrame(data=d1, \
            index=pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05', \
                '2020-01-06', '2020-01-07', '2020-01-10', '2020-01-11', '2020-01-12']))
        self.predict = pd.DataFrame(data=d2, \
            index=pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05', \
                '2020-01-06', '2020-01-07', '2020-01-10', '2020-01-11', '2020-01-12']))
    
        # path_monthly = os.path.join('test','Data','Monthly') 
        # dic_monthly = DP.read_file(path_monthly)

        # n_assets = 1
        # time_series_group = []
        # for i in range(n_assets):
        #     df = dic_monthly[list(dic_monthly.keys())[i]]
        #     ds = DataSeries('ETF', 'monthly', df[0])
        #     time_series_group.append(ds)

        # input_dc = DataCollection('test1', time_series_group)
        # self.input_df = self.input_dc.to_df()

        # train_dc, test_dc = self.input_dc.split(numTest = 12)
        # self.in_sample = train_dc.to_df().rename(columns={"iShares Micro-Cap ETF": "Actual"})
        # actual = test_dc.to_df().rename(columns={"iShares Micro-Cap ETF": "Actual"})

    def test_check_input_df_tickers(self):
        l = []
        l.append(self.actual)
        l.append(self.predict)
        self.assertEqual(MP.check_input_df_tickers(l), True)

        y_naive2_hat_df = pd.DataFrame([False], index = ['Naive2 None'])
        l.append(y_naive2_hat_df)
        self.assertEqual(MP.check_input_df_tickers(l), True)
        
        d = {'bug': [75.0, 72, 69, 70, 76, 36, 74, 78, 72, 62]}
        test = pd.DataFrame(data=d, \
            index=pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05', \
                '2020-01-06', '2020-01-07', '2020-01-10', '2020-01-11', '2020-01-12']))
        l.append(test)
        self.assertEqual(MP.check_input_df_tickers(l), False)


    def test_MAPE(self):
        pct_chg = abs((self.actual - self.predict) / self.actual)
        mape = pct_chg.mean().item()

        ans = MP.MAPE(self.actual, self.predict)
        self.assertAlmostEqual(mape, 0.03431801341798592)
        self.assertAlmostEqual(ans, mape)

    def test_sMAPE(self):
        diff = abs(self.actual - self.predict)
        abs_sum = abs(self.actual) + abs(self.predict)
        smape = (diff / abs_sum).mean().item()

        ans = MP.sMAPE(self.actual, self.predict)
        self.assertAlmostEqual(ans, smape)

    def test_MASE(self):
        d = {'test': [64, 66.0, 62, 69, 70, 73, 71, 74, 71, 72, 72]}
        in_sample = pd.DataFrame(data=d, \
            index=pd.to_datetime(['2019-12-15','2019-12-16','2019-12-17','2019-12-18','2019-12-19', '2019-12-20', \
                '2019-12-21', '2019-12-22', '2019-12-23', '2019-12-24', '2019-12-25']))

        numerator = (abs(self.actual - self.predict).sum() / len(self.actual)).item()
        denominator = (abs(in_sample.diff(periods=4).dropna()).sum() / (len(in_sample) - 4)).item()
        mase = numerator / denominator

        ans = MP.MASE(self.actual, self.predict, in_sample, 4)
        self.assertAlmostEqual(ans, mase)

    def test_R2(self):
        SSR = ((self.actual - self.predict)**2).sum().item()
        SST = ((self.actual - self.actual.mean())**2).sum().item()
        R2 = 1 - SSR/SST
        
        ans = MP.R2(self.actual, self.predict)
        self.assertAlmostEqual(ans, R2)

    def test_RMSE(self):
        numerator = ((self.actual - self.predict)**2).sum()
        denominator = len(self.actual)
        RMSE = np.sqrt(numerator / denominator).item()

        ans = MP.RMSE(self.actual, self.predict)
        self.assertAlmostEqual(ans, RMSE)

    def test_OWA(self):
        d = {'test': [64, 66.0, 62, 69, 70, 73, 71, 74, 71, 72, 72]}
        in_sample = pd.DataFrame(data=d, \
            index=pd.to_datetime(['2019-12-15','2019-12-16','2019-12-17','2019-12-18','2019-12-19', '2019-12-20', \
                '2019-12-21', '2019-12-22', '2019-12-23', '2019-12-24', '2019-12-25']))

        d_naive = {'test': [71, 70.5, 66, 71, 74.5, 72, 76, 71, 65, 64]}
        naive = pd.DataFrame(data=d_naive, \
            index=pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05', \
                '2020-01-06', '2020-01-07', '2020-01-10', '2020-01-11', '2020-01-12']))

        mase_1 = MP.MASE(self.actual, self.predict, in_sample, 2)
        mase_2 = MP.MASE(self.actual, naive, in_sample, 2)
        smape_1 = MP.sMAPE(self.actual, self.predict)
        smape_2 = MP.sMAPE(self.actual, naive)
        owa = ((mase_1/mase_2) + (smape_1/smape_2)) / 2

        ans = MP.OWA(self.actual, self.predict, in_sample, naive, 2)
        self.assertAlmostEqual(ans, owa)
        
    def test_U1(self):
        numerator = np.sqrt(((self.actual-self.predict)**2).sum()/len(self.actual))
        denominator = np.sqrt((self.actual**2).sum()/len(self.actual))+np.sqrt((self.predict**2).sum()/len(self.predict))
        U1 = (numerator/denominator).item()

        ans = MP.Theil_U1(self.actual, self.predict)
        self.assertAlmostEqual(ans, U1)

    def test_U2(self):
        actual_shift = self.actual.shift(1)
        error = self.actual-self.predict
        error_yt = error/actual_shift
        numerator = np.sqrt((error_yt**2).sum())
        diff = self.actual.diff(1)/actual_shift
        denominator = np.sqrt((diff**2).sum())
        U2 = (numerator/denominator).item()

        ans = MP.Theil_U2(self.actual, self.predict)
        self.assertAlmostEqual(ans, U2)

    def test_MP_class(self):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
     
        path_monthly = os.path.join('test','Data','Monthly') 
        dic_monthly = DP.read_file(path_monthly)

        n_assets = 1
        time_series_group = []
        for i in range(n_assets):
            df = dic_monthly[list(dic_monthly.keys())[i]]
            ds = DataSeries('ETF', 'monthly', df[0])
            time_series_group.append(ds)

        input_dc = DataCollection('test1', time_series_group)
        m = ModelESRNN(seasonality = [12], input_size = 4, output_size = 12, device=device)
        train_dc, test_dc = input_dc.split(numTest = 12)

        m.train(train_dc)

        forecast_dc = m.predict(test_dc) 

        # train_dc.to_df().to_csv('insample.csv')
        test_dc.to_df().to_csv('test.csv')
        # forecast_dc.to_df().to_csv('forecast.csv')
        mn = MN.ModelNaive2(2, train_dc)
        naive2_dc = mn.fit_and_generate_prediction(12, 'MS')
        naive2_dc.to_df().to_csv('naive.csv')

        mp = MP.ModelPerformance("test model performance", 2, test_dc, forecast_dc, train_dc, naive2_dc)
        
        mase = MP.MASE(test_dc.to_df(), forecast_dc.to_df(), train_dc.to_df(), 2)
        smape = MP.sMAPE(test_dc.to_df(), forecast_dc.to_df())
        mape = MP.MAPE(mp.y_df, mp.y_hat_df)
        r2 = MP.R2(test_dc.to_df(), forecast_dc.to_df())
        rmse = MP.RMSE(test_dc.to_df(), forecast_dc.to_df())
        owa = MP.OWA(test_dc.to_df(), forecast_dc.to_df(), train_dc.to_df(), naive2_dc.to_df(), 2)
        u1 = MP.Theil_U1(test_dc.to_df(), forecast_dc.to_df())
        u2 = MP.Theil_U2(test_dc.to_df(), forecast_dc.to_df())

        mp.MASE()
        mp.sMAPE()
        mp.MAPE()
        mp.R2()
        mp.RMSE()
        mp.OWA()
        mp.Theil_U1()
        mp.Theil_U2()

        self.assertAlmostEqual(mp.metrics['sMAPE'], smape)
        self.assertAlmostEqual(mp.metrics['MAPE'], mape)
        self.assertAlmostEqual(mp.metrics['R2'], r2)
        self.assertAlmostEqual(mp.metrics['RMSE'], rmse)
        self.assertAlmostEqual(mp.metrics['MASE'], mase)
        self.assertAlmostEqual(mp.metrics['OWA'], owa)
        self.assertAlmostEqual(mp.metrics['Theil_U1'], u1)
        self.assertAlmostEqual(mp.metrics['Theil_U2'], u2)

if __name__ == "__main__":
    unittest.main()