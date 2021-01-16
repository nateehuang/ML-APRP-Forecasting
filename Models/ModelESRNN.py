from ESRNN import ESRNN
from Models.Model import BaseModel
from utils.Data import DataCollection, DataSeries
import pandas as pd
import warnings

class ModelESRNN(BaseModel):
    ''' 
    ESRNN model object. 
    
    The implement of this class will use esrnn package directly. Will add some 
    modifications.

    Attributes:
    -----------
        input: List[DataCollection object]
            list of DataCollection objects including [X_train, y_train,
            X_test, y_test]
        model_trained: ESRNN object
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

    def __init__(self, max_epochs=15, batch_size=64, batch_size_test=128, freq_of_test=-1,
               learning_rate=1e-3, lr_scheduler_step_size=9, lr_decay=0.9,
               per_series_lr_multip=1.0, gradient_eps=1e-8, gradient_clipping_threshold=20,
               rnn_weight_decay=0, noise_std=0.001,
               level_variability_penalty=80,
               testing_percentile=50, training_percentile=49, ensemble=False,
               cell_type='LSTM',
               state_hsize=40, dilations=[[1, 2], [4, 8]],
               add_nl_layer=False, seasonality=[5], input_size=5, output_size=10,
               frequency='D', max_periods=20, random_seed=1,
               device='cpu', root_dir='./'):
        '''
        Inits ModelESRNN Class.
        Args:
            input_data: pandas dataframe
                the pandas dataframe of the whole dataset used
            *args: 
                other parameters including max_epochs/freq_of_test/etc.
        '''
        model = ESRNN(max_epochs=max_epochs, batch_size=batch_size, 
                    batch_size_test=batch_size_test, freq_of_test=freq_of_test,
                    learning_rate=learning_rate, lr_scheduler_step_size=lr_scheduler_step_size, 
                    lr_decay=lr_decay, per_series_lr_multip=per_series_lr_multip, 
                    gradient_eps=gradient_eps, gradient_clipping_threshold=gradient_clipping_threshold,
                    rnn_weight_decay=rnn_weight_decay, noise_std=noise_std,
                    level_variability_penalty=level_variability_penalty,
                    testing_percentile=testing_percentile, training_percentile=training_percentile, 
                    ensemble=ensemble, cell_type=cell_type, 
                    state_hsize=state_hsize, dilations=dilations,
                    add_nl_layer=add_nl_layer, seasonality=seasonality, 
                    input_size=input_size, output_size=output_size,
                    frequency=frequency, max_periods=max_periods, 
                    random_seed=random_seed, device=device, 
                    root_dir=root_dir)
        super().__init__(model)
        
    def data_reformat(self, train_data: DataCollection, fake_date = None):
        '''
        output the data in the format for ESRNN

        Returns
        ------------
        x_train: pd.DataFrame
        y_train: pd.DataFrame
        '''
        super().check_dc(train_data)

        data = pd.DataFrame(columns = ['unique_id', 'ds', 'x', 'y'])

        for s in train_data:
            ts = s.get_ts()
            ticker = s.get_ticker()
            ts['unique_id'] = ticker
            ts['x'] = s.get_category()
            ts = ts.reset_index()
            # for the csv read in dataPreprocessing,
            # the first column should be Date with column name Date
            if fake_date is not None:
                ts['Date'] = fake_date
            ts = ts.rename(columns={'Date':'ds', ticker:'y'})
            data = pd.concat([data, ts])
        data = data.sort_values(by=['unique_id', 'ds']).reset_index()

        X_train = data[['unique_id', 'ds', 'x']].dropna()
        y_train = data[['unique_id', 'ds', 'y']].dropna()
        
        return X_train, y_train

    def to_dc(self, df, pred_label, pred_freq,):
        ''' reformat forecast dataframe output from predict() into DataCollection Obj.

        Args
        ----------
        pred_label: str
            used to label DataCollection
        pred_freq: dict{ticker: str}
            used as freq of each DataSeries
        '''
        ds_lst = []
        for k, v in df.groupby(['x', 'unique_id']):
            category, ticker= k
            ds_df = v[['ds', 'y_hat']]
            ds_df = ds_df.rename(columns={'ds':'Date', 'y_hat':ticker}).set_index('Date')
            ds_lst.append(DataSeries(category, pred_freq[ticker], ds_df))
        dc = DataCollection(pred_label, ds_lst)
        return dc


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
            # store the last date of training data for prediction purpose
            self.train_last_date = train_data.last_date_list()
            X_train, y_train = self.data_reformat(train_data)
            # Fit model
            self.fit(X_train, y_train)
        else:
            warnings.warn("Model was already trained")
        return self.model

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
        # get the real date for prediction data
        real_date = []
        for series in y_dc:
            real_date += series.get_ts().index.to_list()
        # generate fake date
        fake_date = pd.date_range(self.train_last_date[0], periods=len(y_dc[0])+1)
        fake_date = fake_date[1:]
        # reformat as package model required 
        X, _ = self.data_reformat(y_dc, fake_date)
        # Predict on test set
        y_hat_df = self.model.predict(X)
        y_hat_df['ds'] = real_date
        # return type should be a DataCollection object
        y_hat_dc = self.to_dc(y_hat_df, pred_label, pred_freq)
        return y_hat_dc