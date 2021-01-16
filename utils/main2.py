import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
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

def model_ESRNN(train_dc: DataCollection, test_dc: DataCollection, input_size = 5, output_size = 30, 
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

def model_naive2(train_dc: DataCollection, test_dc: DataCollection, numTest: int, seasonality = 5):
    m = ModelNaive2(seasonality, train_dc, test_dc)
    naive_dc = m.fit_and_generate_prediction(numTest,'D')
    # naive_dc.to_df().to_csv('main_naive2_forecast_result.csv')
    return naive_dc

def model_performance(model_name: str, train_dc: DataCollection, 
                    test_dc: DataCollection, forecast_dc: DataCollection, naive2_dc = None):

    MP = ModelPerformance(label= model_name + ' Performance', seasonality = 1, 
                        y_dc = test_dc, y_hat_dc = forecast_dc, 
                        y_insample_dc = train_dc, y_naive2_hat_dc = naive2_dc)
    metrics_dic = MP.generate_all_metrics()
    performance_df = pd.DataFrame.from_dict(metrics_dic, orient='index').rename(columns={0:MP.label})
    print('===== '+ MP.label + ' =====')
    print(performance_df)
    return performance_df, train_dc, test_dc, forecast_dc

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

    # num = int(input('Enter the portfolio number: '))
    num = 2

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
    PP.PnL()

    # PP.print_metrics()
    return (PP.get_metrics('annualized_return'), PP.get_metrics('annualized_volatility'), PP.get_metrics('sharpe_ratio'), PP.get_metrics('max_drawdown'), PP.get_metrics('PnL'))

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
    # create path
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    path_daily = parent_dir + '/Data/Daily'
    # path_daily = '/Users/Zibin/Desktop/Data/Daily'
    path_daily = '/Users/Zibin/Desktop/all_after2005'
    # decide forecast horizon, rolling time
    numTest = 90
    numRoll = 8
    # read whole dataset into a DataCollection
    dc, recover_list, ticker_list = dc_generator(path_daily, 'daily')

    # Obtain DataCollection for Telescope Recommender training
    # rec_dc = dc.trim('2005-01-01','2010-12-31')

    # instantiate a Telescope Model
    m = ModelTelescope()

    # m.train_recommender(rec_dc)

    # Obatin DataCollection for Rolling 
    date = dc.to_df().index # store dates
    
    train_start_index = 0

    # rolling
    rolling_results = []

    ben_return = []
    ben_vol = []
    ben_sharpe = []
    ben_maxdd = []
    ben_pnl = []
    enhance_return = []
    enhance_vol = []
    enhance_sharpe = []
    enhance_maxdd = []
    enhance_pnl = []
    
    for i in range(numRoll):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # m = ModelESRNN(max_epochs = 15, seasonality = [5], batch_size = 5,
        #             input_size = 5, output_size = 90, dilations = [[1,5]], device = device)
        print()
        print()
        print()
        print('=============================================== Rolling '+str(i)+'================================================')
        # train_dc
        train_dc = dc.trim(date[train_start_index], date[train_start_index + 2519])
        m.train(train_dc)
        
        # test_dc
        test_start_index = train_start_index + 2520
        test_dc = dc.trim(date[test_start_index], date[test_start_index + numTest - 1])

        # evaluate_dc
        evaluate_start_index = test_start_index + numTest 
        evaluate_dc = dc.trim(date[evaluate_start_index], date[evaluate_start_index + numTest - 1])
        
        # forecast_dc
        forecast_dc = m.predict(numTest, test_dc) # for telescope
        # forecast_dc = m.predict(test_dc) # for esrnn
        # store forecasts for all of rolling into one df
        if i == 0:
            forecast_results_df = forecast_dc.to_df()
        else: 
            forecast_results_df = forecast_results_df.append(forecast_dc.to_df())
        
        # naive_dc
        naive_dc = model_naive2(train_dc, test_dc, numTest)

        # rolling results in tuples
        rolling_results.append((train_dc, forecast_dc, test_dc, naive_dc))
        train_start_index += numTest

        historical_dc = train_dc.__add__(test_dc)

        forecast_dc = recover_return(forecast_dc.to_df(), recover_list, ticker_list)
        historical_dc = recover_return(historical_dc.to_df(), recover_list, ticker_list)
        evaluate_dc = recover_return(evaluate_dc.to_df(), recover_list, ticker_list)

        # forecast_dc.to_df().to_csv('checkforecast.csv')
        # historical_dc.to_df().to_csv('checkhistorical.csv')
        # evaluate_dc.to_df().to_csv('checkevaluate.csv')

        # tic = ['SPY', 'QQQ', 'iShares Russell 1000 Growth ETF', 'AGG', 'iShares Russell Mid-Cap ETF']

        # historical_dc = historical_dc.select_tickers(tic)
        # forecast_dc = forecast_dc.select_tickers(tic)
        # evaluate_dc = evaluate_dc.select_tickers(tic)

        print('Benchmark:')
        bench = portfolio_performance(historical_dc, evaluate_dc)
        ben_return.append(bench[0])
        ben_vol.append(bench[1])
        ben_sharpe.append(bench[2])
        ben_maxdd.append(bench[3])
        ben_pnl.append(bench[4])

        print('Enhanced:')
        enhance = portfolio_performance(forecast_dc, evaluate_dc)
        enhance_return.append(enhance[0])
        enhance_vol.append(enhance[1])
        enhance_sharpe.append(enhance[2])
        enhance_maxdd.append(enhance[3])
        enhance_pnl.append(enhance[4])

        # print(bench[4])
        # print(enhance[4])
        
        # plt.plot(bench[4].index, bench[4]['PnL'])
        # plt.plot(enhance[4].index, enhance[4]['PnL'])
        # plt.show()

    forecast_results_df.to_csv('ESRNN_recomd_rolling_forecasts.csv')
    comb_MPs = []
    for i, tuples in enumerate(rolling_results):
        MP_df, _, _, _= model_performance(model_name = 'ESRNN Rolling '+ str(i), train_dc = tuples[0], 
                                    test_dc= tuples[2], forecast_dc= tuples[1], naive2_dc = tuples[3])
        comb_MPs.append(MP_df)
    comb_MP_df = pd.concat(comb_MPs, axis=1)
    comb_MP_df['Mean'] = comb_MP_df.mean(axis=1)
    print(comb_MP_df)
    comb_MP_df.to_csv('combMP.csv')

    comb_PPs = []
    d1 = {'Benchmark': [sum(ben_return)/numRoll, sum(ben_vol)/numRoll, sum(ben_sharpe)/numRoll, sum(ben_maxdd)/numRoll]}
    d2 = {'Enhanced': [sum(enhance_return)/numRoll, sum(enhance_vol)/numRoll, sum(enhance_sharpe)/numRoll, sum(enhance_maxdd)/numRoll]}
    comb_PPs.append(pd.DataFrame(data=d1, index = ['annualized return', 'annualized volatility', 'Sharpe ratio', 'Max DD']))
    comb_PPs.append(pd.DataFrame(data=d2, index = ['annualized return', 'annualized volatility', 'Sharpe ratio', 'Max DD']))
    comb_PP_df = pd.concat(comb_PPs, axis=1)
    comb_PP_df.to_csv('combPP.csv')

    comb_PnLs = []
    for i in range(len(enhance_pnl)):
        com_pnl = ben_pnl[i].merge(enhance_pnl[i], how='outer', left_index=True, right_index=True)
        comb_PnLs.append(com_pnl)
    comb_PnL_df = pd.concat(comb_PnLs, axis=1)
    comb_PnL_df.to_csv('combPnL.csv')

    print(ben_sharpe)
    print(enhance_sharpe)

    
