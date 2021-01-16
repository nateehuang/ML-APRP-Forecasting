import unittest             # The test framework
import utils.DataPreprocessing as DataPreprocessing    # The code to test
from pandas.util.testing import assert_frame_equal # <-- for testing dataframes
import pandas as pd
import numpy as np
import os

class Test_DataPreprocessing(unittest.TestCase):
    
    def test_read_file(self):
        
        path_daily = os.path.join('test','Data','Daily')
        #path_daily = r'test\Data\Daily' 
        dic_daily = DataPreprocessing.read_file(path_daily)
        # not whole dataset, only have 3 files including
        # AGG.csv   AA.csv  SP MidCap 400.xls
        
        self.assertTrue(type(dic_daily)==dict)
        assert(isinstance(dic_daily['AGG'][0], pd.DataFrame))
        self.assertTrue(dic_daily['AGG'][1]=='ETF')
        self.assertTrue(dic_daily['AA'][1]=='Stock')
        self.assertEqual(len(dic_daily),3)

        path_monthly = os.path.join('test','Data','Monthly')
        dic_monthly = DataPreprocessing.read_file(path_monthly)
        self.assertTrue(type(dic_monthly)==dict)
        assert(isinstance(dic_monthly['AGG'][0], pd.DataFrame))
        self.assertTrue(dic_monthly['AGG'][1]=='ETF')
        self.assertTrue(dic_monthly['AA'][1]=='Stock')
        self.assertNotEqual(len(dic_daily),0)

        
    def test_simple_imputation(self):
        df = pd.DataFrame([10.2, np.NaN, 32.1, np.NaN], columns=['fakeSPY'],
                        index=pd.to_datetime(['2020-01-01','2020-02-01',
                        '2020-03-01','2020-04-01'])) 
        self.assertEqual(df.isnull().sum().values[0],2)
        new_df = DataPreprocessing.simple_imputation(df)
        self.assertEqual(new_df.isnull().sum().values[0],0)

    def test_csv_to_pd(self):
        single_csv = os.path.join('test','Data','Daily','ETF','AGG.csv')
        #single_csv = r'test\Data\Daily\ETF\AGG.csv'
        df = DataPreprocessing.csv_to_pd(single_csv)
        assert(isinstance(df, pd.DataFrame))
        assert(isinstance(df.index,pd.DatetimeIndex))
        
    def test_excel_to_pd(self):
        single_excel = os.path.join('test','Data','Daily','ETF','SP MidCap 400.xls')
        #single_excel = r'test\Data\Daily\ETF\SP MidCap 400.xls'
        df = DataPreprocessing.excel_to_pd(single_excel)
        assert(isinstance(df, pd.DataFrame))
        assert(isinstance(df.index,pd.DatetimeIndex))

    def test_read_single_file(self):
        single_csv = os.path.join('test','Data','Daily','ETF','AGG.csv')
        #single_csv = r'test\Data\Daily\ETF\AGG.csv'
        single_excel = os.path.join('test','Data','Daily','ETF','SP MidCap 400.xls')
        # single_excel = r'test\Data\Daily\ETF\SP MidCap 400.xls'
        df_csv = DataPreprocessing.read_single_file('AGG.csv',single_csv)
        df_excel = DataPreprocessing.read_single_file('SP MidCap 400.xls',
                                                    single_excel)
        assert(isinstance(df_csv, pd.DataFrame))
        assert(isinstance(df_csv.index,pd.DatetimeIndex))
        assert(isinstance(df_excel, pd.DataFrame))
        assert(isinstance(df_excel.index,pd.DatetimeIndex))

    def test_single_price(self):
        df1 = pd.DataFrame({'Open':[10.2, 12, 32.1, 9.32],'Close':[2.3, 3.6, 4.5, 11.11]},
                        index=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-04-01']))
        expected1 = pd.DataFrame({'Fake_1':[2.3, 3.6, 4.5, 11.11]},
                        index=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-04-01']))
        df2 = pd.DataFrame({'FakeSPY':[10.2, 12, 32.1, 9.32]},
                        index=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-04-01']))
        expected2 = pd.DataFrame({'FakeSPY':[10.2, 12, 32.1, 9.32]},
                        index=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-04-01']))
        new_df1 = DataPreprocessing.single_price(df1,'Fake_1')
        new_df2 = DataPreprocessing.single_price(df2,'FakeSPY')

        assert_frame_equal(new_df1, expected1)
        assert_frame_equal(new_df2, expected2)

if __name__ == '__main__':
    unittest.main()    
