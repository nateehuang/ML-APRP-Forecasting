import unittest
import pandas as pd
import utils.DataPreprocessing as preprocess
from utils.Data import DataSeries, DataCollection
from Models.ModelESRNN import ModelESRNN
import os

'''
Target: try each config set and run to record time used, and pay attention on how is the training loss, 
        need to be not too large not too small (overfit).

From paper,
    non-seasonal: yearly/daily
    signle-seasonal: monthly, quarterly, weekly
    double-seasonal: hourly

    

For esrnn c++:
    string VARIABLE = "Daily";
    const string run = "Final 50/49 730 4/5 (1,3)(7,14) LR=3e-4 {9,1e-4f} EPOCHS=13, LVP=100 13w";
    //#define USE_RESIDUAL_LSTM
    //#define USE_ATTENTIVE_LSTM
    ADD_NL_LAYER = false;
    PERCENTILE = 50; //we always use Pinball loss. When forecasting point value, we actually forecast median, so PERCENTILE=50
    TRAINING_PERCENTILE = 49;  //the program has a tendency for positive bias. So, we can reduce it by running smaller TRAINING_PERCENTILE
    SEASONALITY_NUM = 1; //0 means no seasonality, for Yearly; 1 - single seasonality for Daily(7), Weekly(52); 2 - dual seaonality for Hourly (24,168)
    SEASONALITY = 7;
    SEASONALITY2 = 0;
    dilations = { { 1,3 },{ 7, 14 } };
    INITIAL_LEARNING_RATE = 3e-4;
    LEARNING_RATES = { { 9,1e-4f } }; //at which epoch we manually set them up to what
    PER_SERIES_LR_MULTIP = 1;
    NUM_OF_TRAIN_EPOCHS = 13;
    LEVEL_VARIABILITY_PENALTY = 100;  //Multiplier for L" penalty against wigglines of level vector. 
    C_STATE_PENALTY = 0;
    int STATE_HSIZE = 40;
    int INPUT_SIZE = 7;
    int OUTPUT_SIZE = 14;
    MIN_INP_SEQ_LEN = 0;
    MIN_SERIES_LENGTH = OUTPUT_SIZE + INPUT_SIZE + MIN_INP_SEQ_LEN + 2;  
        //this is compared to n==(total length - OUTPUT_SIZE). 
            Total length may be truncated by LBACK
        //#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
        //##93     323    2940    2357    4197    9919 
    MAX_SERIES_LENGTH = 13 * SEASONALITY + MIN_SERIES_LENGTH;
    TOPN = 4;

For Daily data,
    seasonanlity: None/7/5/252/[combination]     combination-speed

    input_size: multiple of seasonality 252*i
    output_size: larger than 1, multiple of seasonanlity 252*i  how many is enough for cov matrix estimation
        input > < = output, pay attention on if we have enough data observation for each ticker
    output_size = forecast horizon (13 for weekly data)
    input_size = for seasonal data, >= a full season (4 for quarterly data)
                for non-seasonal data, = forecast horizon
    exact size is determined by backtesting
    
    random_seed: try different, replicate

    all_nl_layer: T/F
    learning_rate: try different value
    step_size: initial step size, if too large may not converge
    lr_decay: decay speed for step size, avoid to not converge at the gradient point
    epoch, batch_size: if we have 120 data
        SGD -> batch_size LeCun: ideal is 1, highest speed to converge, but in practice we need to consider time
        when batch_size=1, for each epoch, we only use 1 point and then adjust params once, total adjustment is 120
        when batch_size=2, for each epoch, we use 2 point and then adjust param once, total adjustment is 60
        Since GPU can do parallel computation, LeCun: optimal batch_size is 32 when large sample.
        Cannot be 32 when we only have small sample size, for example we only have 32 points, then one epoch use 32 points and adjust once.
    
    A larger batch size enables using a large learning rate. 
    Larger batch sizes tend to have low early losses while training 
    whereas the final loss values are low when the batch size is reduced.

    other hyperparam: gap size for gap Cross Validation          
'''

def run_ESRNN():
        
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        path_daily = r'C:\Users\xxxli\Desktop\Daily'
        dic_daily = preprocess.read_file(path_daily)
        series_list = []
        for k, v in dic_daily.items():
            ticker_name = k
            df, cat = v
            df = preprocess.single_price(df, ticker_name) # column = [ticker]
            series_list.append(DataSeries(cat, 'daily', df))
        collect = DataCollection('universe daily', series_list)
        train_dc, test_dc = collect.split(numTest = 24)

        m = ModelESRNN( max_epochs = 15, 
                    batch_size = 32, dilations=[[1,3], [7, 14]],
                    input_size = 12, output_size = 24, 
                    device = device)

        m.train(train_dc)
        
        y_test = m.predict(test_dc)
        
        y_test_df = y_test.to_df()
        y_test_df.to_csv('hyper_ESRNN_1.csv')

if __name__ == "__main__":
    run_ESRNN()