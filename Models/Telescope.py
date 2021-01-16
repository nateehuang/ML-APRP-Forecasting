from rpy2 import robjects
from rpy2.robjects import FloatVector, ListVector
import rpy2.robjects.packages as rpackages
import pandas as pd
from Models.Model import BaseModel
from utils.Data import DataSeries, DataCollection

def install_telescope():
    '''
    This function has not been tested and not guarantee to work!
    '''
    from rpy2.robjects import StrVector
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind = 1)
    pack = ('devtools', 'remotes')
    utils.install_packages(StrVector(pack))
    rpy2.robjects.r('remotes::install_url(url="https://github.com/DescartesResearch/telescope/archive/master.zip", INSTALL_opt= "--no-multiarch")') 

class ModelTelescope(BaseModel):
    ''' 
    Telescope model object. 
    
    Attributes:
    -----------
        input: List[DataCollection object]
            list of DataCollection objects including [X_train, y_train,
            X_test, y_test]
        model_trained: Telescope object
            the model trained (after model.fit)
        forecasts: pd.DataFrame
            final forecasts of this model
        performances: dict{str: float}
            metrics
        model_name: str
            name of the model
    Methods:
    ---------
        __init__():
        train():
        predict():
            return data object
        evaluate():
    '''

    def __init__(self,):
        '''
        '''
        self.tel = rpackages.importr('telescope')
        self.rec_model = None
        self.freq_map = {'daily': 'D',
                        # weekly should match with specific day in week, i.e., 'W-MON' match every Monday
                        # 'weekly': 'W', 
                        'monthly': 'MS', 
                        'quarterly': 'QS', 
                        'yearly': 'YS'}
    
    def data_reformat(self, train_data: DataCollection):
        '''
        Store information needed
        '''
        super().check_dc(train_data)
        # record information of data
        self.tickers = train_data.ticker_list()
        self.categories = train_data.category_list()
        self.last_days = train_data.last_date_list()
        self.data = train_data.to_list()
        self.label = str(train_data)
        # TODO: double check frequency
        self.frequency = train_data.freq
        self.freq = self.freq_map[train_data.freq]

    def to_dc(self, ):
        pass 

    def train(self, train_data: DataCollection):
        self.data = train_data
        self.data_reformat(train_data)

    def train_recommender(self, train_data: DataCollection):
        rts = robjects.r('ts')
        rts_dict = {ds.get_ticker():rts(FloatVector(ds.get_ts().values), frequency = ds.freq_to_int()) for ds in train_data}
        rlist = ListVector(rts_dict)
        self.rec_model = self.tel.telescope_trainrecommender(rlist)
    
    def predict(self, numPredict:int, test_dc: DataCollection):
        # if recommend model is trained, use recommend model,
        # else do not use recommend model
        if not self.rec_model:
            rec = robjects.r('NULL')
        else:
            rec = self.rec_model
        date = test_dc.to_df().index
        res = []
        for i, series in enumerate(self.data):
            rList = FloatVector(series)
            pred = pd.DataFrame(self.tel.telescope_forecast(rList, 
                                                    numPredict, 
                                                    rec_model = rec, 
                                                    natural = True, 
                                                    boxcox = True, 
                                                    doAnomDet = False, 
                                                    replace_zeros = True, 
                                                    use_indicators = True,
                                                    plot = False)[0],
                                columns = [self.tickers[i]],
                                index = date)
            ds = DataSeries(self.categories[i], self.frequency, pred)
            res.append(ds)
        dc = DataCollection(self.label, res)
        return dc

