import unittest
import pandas as pd
from utils.Data import DataSeries, DataCollection


class Test_Data(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.a = pd.DataFrame([10.2, 12, 32.1, 9.32], columns=['fakeSPY'],
                        index=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-04-01']))
        self.a_series = DataSeries('ETF', 'monthly', self.a)
        self.b = pd.DataFrame([2.3, 3.6, 4.5], columns=['fakeTreasury'],
                            index=pd.to_datetime(['2019-12-12', '2020-02-05', '2020-09-13']))
        self.b_series = DataSeries('Bond', 'monthly', self.b)
        self.c_collection = DataCollection('trial', [self.a_series, self.b_series]) 
        
        # For test_the_rest_of_entire_dataset():
        self.a_entire = pd.DataFrame([10.2, 12, 32.1, 9.32, 11.5, 9.7], columns=['fakeSPY'],
                        index=pd.to_datetime(['2020-01-01','2020-02-01','2020-03-01','2020-04-01','2020-05-01', '2020-06-01']))
        self.a_series_entire = DataSeries('ETF', 'monthly', self.a_entire)
        self.b_entire = pd.DataFrame([2.3, 3.6, 4.5, 5.5], columns=['fakeTreasury'],
                            index=pd.to_datetime(['2019-12-12', '2020-02-05', '2020-09-13','2020-10-13']))
        self.b_series_entire = DataSeries('Bond', 'monthly', self.b_entire)
        self.c_collection_entire = DataCollection('trial', [self.a_series_entire, self.b_series_entire]) 


        self.a_exp = pd.DataFrame([11.5, 9.7], columns=['fakeSPY'],
                        index=pd.to_datetime(['2020-05-01', '2020-06-01']))
        self.a_series_exp = DataSeries('ETF', 'monthly', self.a_exp)
        self.b_exp = pd.DataFrame([5.5], columns=['fakeTreasury'],
                            index=pd.to_datetime(['2020-10-13']))
        self.b_series_exp = DataSeries('Bond', 'monthly', self.b_exp)
        self.c_collection_exp = DataCollection('trial', [self.a_series_exp, self.b_series_exp]) 

    def test_DataSeries_basic(self):
        a = self.a
        a_series = self.a_series
        assert(len(a_series) == 4)
        assert(str(a_series) == 'monthly fakeSPY')
        assert(a_series.get_ticker() == 'fakeSPY')
        assert(a_series.get_category() == 'ETF')
        assert(a_series.get_freq() == 'monthly')
        assert(a.equals(a_series.get_ts()))
        
        # test deep copy
        a_copy = a_series.copy()
        assert(a_copy != a_series and a_copy.get_ts().equals(a_series.get_ts()))
        
        assert(isinstance(a_series.to_Series(), pd.Series))
    
    def test_DataSeries_add_sub(self):
        diff = self.a_series_entire - self.a_series
        assert(self.compareSeries(diff, self.a_series_exp))
        a_plus = diff + self.a_series
        assert(self.compareSeries(a_plus, self.a_series_entire))

    def test_DataSeries_to_list(self):
        lst = self.a_series.to_list()
        assert(lst == [10.2, 12, 32.1, 9.32])

    def test_last_index(self):
        assert(self.a_series.get_last_date() == pd.to_datetime('2020-04-01'))
    
    def test_DataSeries_split_and_trim(self):
        # test split
        a_train, a_test = self.a_series.split(pct = 0.75)
        assert(isinstance(a_train, DataSeries))
        assert(isinstance(a_test, DataSeries))
        assert(len(a_train) == 3)
        assert(len(a_test) == 1)
        assert(self.a.iloc[:3].equals(a_train.get_ts()))
        assert(self.a.iloc[3:].equals(a_test.get_ts()))

        # test trim
        trimed = self.a_series.trim('2020-02-01', '2020-03-01')
        assert(len(trimed) == 2)
        assert(self.a.loc['2020-02-01':'2020-03-01'].equals(trimed.get_ts()))
    
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

    def test_DataCollection_basic(self):
        assert(len(self.c_collection) == 2)
        assert(self.c_collection.get_freq() == 'monthly')
        for item, compare in zip(self.c_collection, [self.a_series, self.b_series]):
            assert(self.compareSeries(item, compare))

    def test_DataCollection_add_sub(self):
        res = self.c_collection_entire - self.c_collection
        expected = self.c_collection_exp
        for r, e in zip(res, expected):
            assert(self.compareSeries(r, e))
        res_plus = res + self.c_collection
        for r, e in zip(res_plus, self.c_collection_entire):
            assert(self.compareSeries(r, e))

    def test_DataCollection_get_series(self):
        item1 = self.c_collection[1]
        assert(self.compareSeries(item1, self.b_series))

        item2 = self.c_collection.get_series('fakeSPY')
        assert(self.compareSeries(item2, self.a_series))

    def test_DataCollection_copy(self):
        c = self.c_collection.copy()
        assert(c != self.c_collection)
        assert(c.label == self.c_collection.label)
        assert(c.get_freq() == self.c_collection.get_freq())
        for one, two in zip(c, self.c_collection):
            assert(self.compareSeries(one, two))
    
    def test_DataCollection_summary(self):
        pass

    def test_DataCollection_split(self):

        train, test = self.c_collection.split(pct = 0.75)
        assert(str(train) == 'trial')
        assert(train.freq == 'monthly')
        assert(str(test) == 'trial')
        assert(test.freq == 'monthly')

        compare = [self.a_series.split(0.75), self.b_series.split(0.75)]
        compare_train, compare_test = zip(*compare)
        train_col, test_col = list(compare_train), list(compare_test)
        for i, item in enumerate(train):
            assert(self.compareSeries(item, train_col[i]))
        
        for i, item in enumerate(test):
            assert(self.compareSeries(item, test_col[i]))

    def test_DataCollection_list(self):
        assert(self.c_collection.ticker_list() == ['fakeSPY', 'fakeTreasury'])
        assert(self.c_collection.category_list() == ['ETF', 'Bond'])
        assert(self.c_collection.last_date_list() == pd.to_datetime(['2020-04-01', '2020-09-13']).to_list())
        assert(self.c_collection.to_list() == [[10.2, 12, 32.1, 9.32], [2.3, 3.6, 4.5]] )

    def test_DataCollection_add(self):
        d = pd.DataFrame([11, 22], columns=['fakeZZZ'],
                            index=pd.to_datetime(['2019-1-12', '2019-02-05']))
        d_series = DataSeries('Bond', 'monthly', d)
        c_plus = self.c_collection.copy()
        c_plus.add(d_series)

        compare = [self.a_series, self.b_series, d_series]
        for i, item in enumerate(c_plus):
            assert(self.compareSeries(item, compare[i]))

    def test_DataCollection_df(self):
        df = self.c_collection.to_df()
        compare = pd.concat([self.a, self.b], axis = 1)
        assert(df.equals(compare))

    def test_price_to_return(self):
        pass

if __name__ == "__main__":
    unittest.main()