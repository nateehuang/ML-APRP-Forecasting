from ESRNN.utils_evaluation import Naive2
from Models.ModelESRNN import ModelESRNN
from utils.Data import DataCollection, DataSeries
from Models.ModelESRNN import ModelESRNN
import numpy as np
import pandas as pd

class ModelNaive2(Naive2, ModelESRNN):
    """
    Naive2 model.
    Popular benchmark model for time series forecasting that automatically adapts
    to the potential seasonality of a series based on an autocorrelation test.
    If the series is seasonal the model composes the predictions of Naive and SeasonalNaive,
    else the model predicts on the simple Naive.

    """
    def __init__(self, seasonality: int, y_insample_dc: DataCollection, test_dc: DataCollection):
        # reformated into esrnn 
        self.X_train_df = self.data_reformat(y_insample_dc)[0]
        self.y_train_df = self.data_reformat(y_insample_dc)[1]
        self.test_dc = test_dc
        # before fit
        self.fitted = False
        # store the frequencies for turning df back to dc
        self.pred_freq = {series.get_ticker():series.get_freq() for series in y_insample_dc} 
        # store the label for turning df back to dc
        self.pred_label = str(y_insample_dc)

        super().__init__(seasonality)

    def fit_and_generate_prediction(self, h: int, freq: str):
        '''
        Args
        ----------
        h: int
            forecasting horizon, should be the same as testing y_df in ModelPerformance
        freq: str
            'H','D','W','MS','QS','D','YS'
        Returns
        ----------
        y_naive2_hat_df: pd.DataFrame
            index: dates
            columns: tickers, should be the same as y_df, y_hat_df, y_insample_df

        '''
        # Naive2 frame
        y_naive2_hat_df = pd.DataFrame(columns = ['unique_id', 'ds', 'x', 'y_hat'])
        # training set we use
        y_train_df = self.y_train_df
        X_train_df = self.X_train_df
        # Sort training set by unique_id for faster loop
        y_train_df = y_train_df.sort_values(by = ['unique_id', 'ds'])
        # List of uniques ids
        unique_ids = y_train_df['unique_id'].unique()
        # Panel of fitted models
        
        dates_from_testing = self.test_dc.to_df().index
        for unique_id in unique_ids:
            # Fast filter y by id.
            top_row = np.asscalar(y_train_df['unique_id'].searchsorted(unique_id, 'left'))
            bottom_row = np.asscalar(y_train_df['unique_id'].searchsorted(unique_id, 'right'))
            y_id = y_train_df[top_row:bottom_row]
            # for each series
            y_naive2 = pd.DataFrame(columns = ['unique_id', 'ds', 'x', 'y_hat'])
            y_naive2['ds'] = dates_from_testing
            '''
            y_naive2['ds'] = pd.date_range(start = y_id.ds.max(),
                                        periods = h + 1, freq = freq)[1:]
            '''
            y_naive2['unique_id'] = unique_id
            y_naive2['y_hat'] = self.fit(y_id.y.to_numpy()).predict(h)
            y_naive2['x'] = X_train_df.loc[X_train_df['unique_id'] == unique_id, 'x'].tolist()[0]

            y_naive2_hat_df = y_naive2_hat_df.append(y_naive2)
            
        # y_naive2_hat_df.to_csv('model_y_naive2_hat_df.csv')

        res = self.to_dc(y_naive2_hat_df, self.pred_label, self.pred_freq)
        self.fitted = True
        return res