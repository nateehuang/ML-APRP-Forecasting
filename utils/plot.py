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
    dic = DataPreprocessing.read_file(path)
    series_list = []
    for k, v in dic.items():
        df, cat = v
        df = DataPreprocessing.single_price(df, k)
        series_list.append(DataSeries(cat, frequency, df))
    collect = DataCollection(frequency + ' Collection', series_list)
    return collect

def model_ESRNN(train_dc: DataCollection, test_dc: DataCollection, input_size = 5, output_size = 252, num_epoch = 15, batch_size = 64, dilations = [[1,5]], seasonality = [5,7]):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    m = ModelESRNN(max_epochs = num_epoch, seasonality = seasonality, batch_size = batch_size,
                        input_size = input_size, output_size = output_size, dilations = dilations, device = device)

    m.train(train_dc)
    forecast_dc = m.predict(test_dc)
    forecast_df = forecast_dc.to_df()
    forecast_df.to_csv('main_esrnn_predict_result.csv')
    return train_dc, test_dc, forecast_dc

def model_telescope(recomd: bool, train_dc: DataCollection, test_dc: DataCollection, numTest: int, seasonality = 5):

    m = ModelTelescope()
    if recomd:
        m.train_recommender(train_dc)
    m.train(train_dc)
    forecast_dc = m.predict(numTest, test_dc)
    forecast_dc.to_df().to_csv(str(recomd)+'_main_telescope_result.csv')
    return train_dc, test_dc, forecast_dc

def model_telescope(train_dc: DataCollection, test_dc: DataCollection, numTest: int, seasonality = 5):

    m = ModelTelescope()
    # m.train_recommender(train_dc)
    m.train(train_dc)
    forecast_dc = m.predict(numTest)
    forecast_dc.to_df().to_csv('telescope_result.csv')

    return train_dc, test_dc, forecast_dc

path_daily = '/Users/Zibin/Desktop/Data/Daily'

numTest = 42
dc = dc_generator(path_daily, 'daily')
input_dc, portfolio_dc = dc.split(numTest = numTest)
train_dc, test_dc = input_dc.split(numTest = numTest)


# test_df: forecast horizon h
test_df = test_dc.to_df()
test_df_AGG = test_df['AGG']
h = len(test_df_AGG.to_numpy()) # 30
print('=== test df AGG ===')
print(test_df_AGG) #2019-10-07 2019-11-15

# training data
train_df_AGG = train_dc.to_df()['AGG']
print('=== train df AGG ===')
print(train_df_AGG) #2011-01-03 2019-10-04

# naive2
print('=== naive df AGG ===')
m = ModelNaive2(5, train_dc, test_dc)
naive_y_hat_dc = m.fit_and_generate_prediction(h,'D')

naive_y_hat_dc.to_df().to_csv('main_new_naive.csv')
naive_y_hat_AGG = naive_y_hat_dc.to_df()['AGG']
print(naive_y_hat_AGG)

# esrnn
print('=== ESRNN df AGG ===')
_, _, forecast_dc = model_ESRNN(train_dc,test_dc,output_size=h)
forecast_df = forecast_dc.to_df()
print(forecast_df['AGG'])

print('=== MP ESRNN ===')
MP = ModelPerformance('ESRNN Performance', 1, test_dc, forecast_dc, train_dc, naive_y_hat_dc)
metrics_dic = MP.generate_all_metrics()
performance_df = pd.DataFrame.from_dict(metrics_dic, orient='index').rename(columns={0:'metrics'})
print(performance_df)

# telescope without recommender
print('=== Telescope no recomd df AGG ===')
_, _, forecast_dc = model_telescope(False,train_dc,test_dc,numTest,5)
forecast_telescope_df = forecast_dc.to_df()
print(forecast_telescope_df['AGG'])

print('=== MP Telescope No Recomd ===')
MP = ModelPerformance('Telescope', 1, test_dc, forecast_dc, train_dc, naive_y_hat_dc)
metrics_dic = MP.generate_all_metrics()
performance_df = pd.DataFrame.from_dict(metrics_dic, orient='index').rename(columns={0:'metrics'})
print(performance_df)

# telescope with recommender
print('=== Telescope w recomd df AGG ===')
_, _, forecast_dc = model_telescope(True,train_dc,test_dc,numTest,5)
forecast_telescope_recomd_df = forecast_dc.to_df()
print(forecast_telescope_recomd_df['AGG'])

MP = ModelPerformance('Telescope', 1, test_dc, forecast_dc, train_dc, naive_y_hat_dc)
metrics_dic = MP.generate_all_metrics()
performance_df = pd.DataFrame.from_dict(metrics_dic, orient='index').rename(columns={0:'metrics'})
print(performance_df)

# plot
print('=== plot ===')
plt.plot(forecast_df['AGG'], label='esrnn predict')
plt.plot(forecast_telescope_df['AGG'], label='telescope no recomd')
plt.plot(forecast_telescope_recomd_df['AGG'], label='telescope recomd')
plt.plot(test_df_AGG, label='real testing data')
plt.plot(naive_y_hat_AGG, label='naive predict')
plt.title('AGG')
plt.legend()
plt.show()