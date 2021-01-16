#######################################
import os
import torch
import pandas as pd
import utils.Portfolio as port
import utils.DataPreprocessing as DataPreprocessing
from utils.Data import DataSeries, DataCollection
from Models.ModelESRNN import ModelESRNN
from Models.Telescope import ModelTelescope
from Models.ModelNaive2 import ModelNaive2
from utils.Model_Performance import ModelPerformance
from utils.Portfolio_Performance import PortfolioPerformance 
import matplotlib.pyplot as plt 

def dc_generator(path: str, frequency: str):
    dic, recover_list, ticker_list = DataPreprocessing.read_file(path)
    series_list = []
    for k, v in dic.items():
        df, cat = v
        df = DataPreprocessing.single_price(df, k)
        series_list.append(DataSeries(cat, frequency, df))
    collect = DataCollection(frequency + ' Collection', series_list)
    return collect, recover_list, ticker_list

def model_ESRNN(train_dc: DataCollection, test_dc: DataCollection, input_size = 5, output_size = 90, 
                num_epoch = 15, batch_size = 5, dilations = [[1,5]], seasonality = [5]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    m = ModelESRNN(max_epochs = num_epoch, seasonality = seasonality, batch_size = batch_size,
                    input_size = input_size, output_size = output_size, dilations = dilations, device = device)

    m.train(train_dc)
    forecast_dc = m.predict(test_dc) 
    forecast_df = forecast_dc.to_df()
    forecast_df.to_csv('main_esrnn_predict_result.csv')

    return train_dc, test_dc, forecast_dc

def model_telescope(recomd: bool, train_dc: DataCollection, test_dc: DataCollection, numTest: int):

    m = ModelTelescope()
    if recomd:
        m.train_recommender(train_dc)
    m.train(train_dc)
    forecast_dc = m.predict(numTest, test_dc)
    # forecast_dc.to_df().to_csv(str(recomd)+'_main_telescope_result.csv')

    return train_dc, test_dc, forecast_dc

def model_naive2(train_dc, test_dc, numTest: int, seasonality = 5):
    m = ModelNaive2(seasonality, train_dc, test_dc)
    naive_dc = m.fit_and_generate_prediction(numTest,'D')
    # naive_dc.to_df().to_csv('main_naive2_forecast_result.csv')
    return naive_dc

def model_performance(model_name: str, train_dc: DataCollection, test_dc: DataCollection, 
                        numTest: int, naive_dc=None, telescope_recomd=False):
    if model_name.upper() == 'ESRNN':
        train, test, forecast = model_ESRNN(train_dc, test_dc, output_size=numTest)
    
    elif model_name.upper() == 'TELESCOPE':
        train, test, forecast = model_telescope(telescope_recomd,train_dc, test_dc, numTest)

    else:
        raise NameError('Such model does not exist!')
    
    MP = ModelPerformance(label=model_name + ' Performance', seasonality=1, y_dc = test, y_hat_dc=forecast,
                         y_insample_dc=train, y_naive2_hat_dc = naive_dc)
    
    metrics_dic = MP.generate_all_metrics()

    performance_df = pd.DataFrame.from_dict(metrics_dic, orient='index').rename(columns={0:'metrics'})
    print(performance_df)
    return performance_df, train, test, forecast, naive_dc

def recover_return(input_df: pd.DataFrame, recover_list, ticker_list):
    # input_df = input_df + recover - 1
    ds_list = []
    for column in input_df:
        idx = ticker_list.index(column)
        recover_num = recover_list[idx]
        temp_series = input_df[column] + recover_num - 1
        ds_list.append(DataSeries('ETF', 'daily', temp_series.to_frame()))
    output_dc = DataCollection('Daily Collection', ds_list)
    return output_dc

def single_ticker_return(ticker, forecast_df, test_df, naive_df):
    test = test_df[ticker]
    forecast = forecast_df[ticker]
    naive = naive_df[ticker]
    
    tick = [naive.index[i] for i in range(0,len(naive),22)]
    plt.figure(figsize=(9,6))
    plt.plot(test)
    plt.plot(forecast)
    plt.plot(naive)
    plt.xticks(tick)
    plt.legend(['Actual', 'Forecast', 'Benchmark'])
    plt.show()

if __name__ == '__main__':

    # parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    # path_daily = parent_dir + '/Data/Daily'

    # path_daily = '/Users/Zibin/Desktop/all_after2005'

    # numTest = 90
    # dc, recover_list, ticker_list = dc_generator(path_daily, 'daily')
    # input_dc, evaluate_dc = dc.split(numTest = numTest)
    # train_dc, test_dc = input_dc.split(numTest = numTest)
    # # evaluate_dc.to_df().to_csv('evaluate.csv')

    # input_dc.to_df().to_csv('input.csv')

    # naive_dc = model_naive2(train_dc, test_dc, numTest)
    # # performance_ESRNN, _, _, forecast, naive = model_performance('ESRNN', train_dc, test_dc,  numTest, naive_dc)
    # performance_TELESCOPE, _, _, forecast, naive = model_performance('TELESCOPE', train_dc, test_dc, numTest, naive_dc, False)

    # forecast_dc = recover_return(forecast.to_df(), recover_list, ticker_list)
    # test_dc = recover_return(test_dc.to_df(), recover_list, ticker_list)
    # naive_dc = recover_return(naive.to_df(), recover_list, ticker_list)
    # forecast_dc.to_df().to_csv('telescope_forecast.csv')
    # test_dc.to_df().to_csv('telescope_test.csv')
    # naive_dc.to_df().to_csv('telescope_naive.csv')

    forecast_df = pd.read_csv('telescope_forecast.csv', index_col=['Date'])
    test_df = pd.read_csv('telescope_test.csv', index_col=['Date'])
    naive_df = pd.read_csv('telescope_naive.csv', index_col=['Date'])

    single_ticker_return('AGG', forecast_df, test_df, naive_df)