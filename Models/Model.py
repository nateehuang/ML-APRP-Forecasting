from utils.Data import DataCollection
import warnings
import numpy as np
from abc import abstractmethod

class BaseModel(object):
    '''
    Base Class for all the models:

    Attributes
    ----------
    self.fitted: bool
    self.model: model initiated
    
    '''

    def __init__(self, model):
        self.fitted = False
        self.model = model

    @staticmethod
    def check_dc(dc: DataCollection):
        '''
        Check data is a DataCollection object

        '''
        if not isinstance(dc, DataCollection):
            raise TypeError('Input train_data should be a DataCollection object')
    
    @abstractmethod
    def data_reformat(self, data):
        raise NotImplementedError()

    @abstractmethod
    def to_dc(self, df, pred_label, pred_freq):
        raise NotImplementedError

    def train(self, train_data: DataCollection):
        ''' 
        Model train: fit & validation

        Args
        ----------
        train_data: DataCollection
            training set, y_insample_dc

        Returns:
        ----------
            the trained model
            
        '''
        if not self.fitted:
            X_train, y_train = self.data_reformat(train_data)
            # Fit model
            self.fit(X_train, y_train)
        else:
            warnings.warn("Model was already trained")
        return self.model

    def fit(self, X_train, y_train):
        ''' 
        Model train: fit & validation
        Args
        ----------
        X_train: pd.DataFrame
        y_train: pd.DataFrame
            formatted as package model required

        Returns:
        ----------
            the trained model
            
        '''
        # Fit model
        self.model.fit(X_train, y_train)
        self.fitted = True
    
    def predict(self, y_dc: DataCollection):
        ''' Predict on test set.

        The function uses the trained model to do forecast on our test set.

        Args
        ----------
            y_dc: DataCollection
                testing data, OOS

        Returns
        ----------
            y_hat_dc: DataCollection

        '''
        # store the frequencies for turning df back to dc
        pred_freq = {series.get_ticker():series.get_freq() for series in y_dc} 
        # store the label for turning df back to dc
        pred_label = str(y_dc)
        # reformat as package model required 
        X, _ = self.data_reformat(y_dc)
        # Predict on test set
        y_hat_df = self.model.predict(X)
        # return type should be a DataCollection object
        y_hat_dc = self.to_dc(y_hat_df, pred_label, pred_freq)
        return y_hat_dc

    # def validation_simple(self, input_dc: DataCollection, split_pct = 0.8):
    #     train_dc, validation_dc = input_dc.split(pct = split_pct)

    #     validation_df = validation_dc.to_df()

    #     self.train(train_dc)
    #     y_predict = self.predict(validation_dc) 
    #     y_predict_df = y_predict.to_df()

    #     diff = y_predict_df - validation_df
    #     rmse_sum = 0
    #     for i in range(len(diff.columns)):
    #         rmse_sum += np.sqrt(((diff.iloc[:,i].dropna())**2).mean())
    #     score = rmse_sum / len(diff.columns)
        
    #     return score

    # def validation_rolling(self, input_dc: DataCollection, split_pct: float, num_split: int, epochs: int, batch_size: int, input_size: int, output_size: int):
    #     train_val_dic = {}
    #     for i in range(num_split):
    #         train, validation = input_dc.split(pct = split_pct)
    #         train_val_dic[i] = [train, validation]
    #         input_dc = train

    #     total_score = 0
    #     for i in range(num_split-1, -1, -1):
    #         train_dc = train_val_dic[i][0]
    #         validation_dc = train_val_dic[i][1]

    #         validation_df = validation_dc.to_df()
    #         device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #         m = ModelESRNN(max_epochs = epochs, seasonality = [], batch_size = batch_size, input_size = input_size, output_size = output_size, device = device)
    #         m.train(train_dc)
    #         y_predict = m.predict(validation_dc) 
    #         y_predict_df = y_predict.to_df()

    #         diff = y_predict_df - validation_df
    #         rmse_sum = 0
    #         for i in range(len(diff.columns)):
    #             rmse_sum += np.sqrt(((diff.iloc[:,i].dropna())**2).mean())
    #         total_score += rmse_sum / len(diff.columns)

    #     score = total_score / num_split

    #     return score
