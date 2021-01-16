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
    return performance_df, train, test, forecast

def portfolio_generator(input_dc: DataCollection):
    solution = {}
    solution[0] = 'equal_portfolio'
    solution[1] = 'max_sharpe'
    solution[2] = 'min_volatility'
    solution[3] = 'efficient_return'
    solution[4] = 'efficient_risk'
    solution[5] = 'black_litterman_portfolio'

    print('To construct an equal weighted portfolio,   enter 0')
    print('To construct a max sharpe ratio portfolio,  enter 1')
    print('To construct a minimum variance portfolio,  enter 2')
    print('To construct an efficient return portfolio, enter 3')
    print('To construct an efficient risk portfolio,   enter 4')
    print('To construct a black litterman portfolio,   enter 5')

    num = int(input('Enter the portfolio number: '))

    if num == 0:
        p = port.EqualPort(solution[0])
        p.calculate_initial_weight(input_dc)
    elif num == 1:
        p = port.MaxSharpePort(solution[1])
        p.calculate_initial_weight(input_dc)
    elif num == 2:
        p = port.MinVolPort(solution[2])
        p.calculate_initial_weight(input_dc)
    elif num == 3:
        p = port.EffReturnPort(solution[3])
        p.calculate_initial_weight(input_dc)
    elif num == 4:
        p = port.EffRiskPort(solution[4])
        p.calculate_initial_weight(input_dc)
    elif num == 5:
        p = port.BlackLittermanPort(solution[5])
        p.calculate_initial_weight(input_dc)
    
    return p

def portfolio_performance(input_dc: DataCollection, evaluate_dc: DataCollection):
    p = portfolio_generator(input_dc)
    PP = PortfolioPerformance(p, evaluate_dc)

    PP.annualized_return()
    PP.annualized_volatility()
    PP.annualized_sharpe_ratio()
    PP.max_drawdown()
    PP.omega_ratio()
    PP.sortino_ratio()

    PP.print_metrics()

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

if __name__ == '__main__':

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    path_daily = parent_dir + '/Data/Daily'

    path_daily = '/Users/Zibin/Desktop/all_after2005'

    numTest = 90
    dc, recover_list, ticker_list = dc_generator(path_daily, 'daily')
    input_dc, evaluate_dc = dc.split(numTest = numTest)
    train_dc, test_dc = input_dc.split(numTest = numTest)
    evaluate_dc.to_df().to_csv('evaluate.csv')

    input_dc.to_df().to_csv('input.csv')
    
    naive_dc = model_naive2(train_dc, test_dc, numTest)
    performance_ESRNN, train, test, forecast = model_performance('ESRNN', train_dc, test_dc,  numTest, naive_dc)
    # performance_TELESCOPE, train, test, forecast = model_performance('TELESCOPE', train_dc, test_dc, numTest, naive_dc, False)

    forecast_dc = recover_return(forecast.to_df(), recover_list, ticker_list)
    historical_dc = recover_return(input_dc.to_df(), recover_list, ticker_list)
    evaluate_dc = recover_return(evaluate_dc.to_df(), recover_list, ticker_list)

    # tic = ['SPY', 'QQQ', 'iShares Russell 2000 Value ETF', 'XLF', 'iShares Russell 2000 Growth ETF', 'iShares Russell Mid-Cap ETF']

    # historical_dc = historical_dc.select_tickers(tic)
    # forecast_dc = forecast_dc.select_tickers(tic)
    # evaluate_dc = evaluate_dc.select_tickers(tic)

    # forecast_dc.to_df().to_csv('checkforecast.csv')
    # historical_dc.to_df().to_csv('checkhistorical.csv')
    # evaluate_dc.to_df().to_csv('checkevaluate.csv')

    print('Benchmark:')
    portfolio_performance(historical_dc, evaluate_dc)

    print('Enhanced:')
    portfolio_performance(forecast_dc, evaluate_dc)





