import os
import torch
import pandas as pd
import numpy as np
from utils.Data import DataCollection, DataSeries
from Models.ModelESRNN import ModelESRNN
import utils.DataPreprocessing as preprocess
import utils.Model_Performance as MP

def validation_simple(input_dc: DataCollection, numTest: int,
                    max_epochs=15, batch_size=1, batch_size_test=128, freq_of_test=-1,
                    learning_rate=1e-3, lr_scheduler_step_size=9, lr_decay=0.9,
                    per_series_lr_multip=1.0, gradient_eps=1e-8, gradient_clipping_threshold=20,
                    rnn_weight_decay=0, noise_std=0.001,
                    level_variability_penalty=80,
                    testing_percentile=50, training_percentile=50, ensemble=False,
                    cell_type='LSTM',
                    state_hsize=40, dilations=[[1, 2], [4, 8]],
                    add_nl_layer=False, seasonality=[4], input_size=4, output_size=8,
                    frequency=None, max_periods=20, random_seed=1,):
    train_dc, validation_dc = input_dc.split(numTest = numTest)

    validation_df = validation_dc.to_df()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = ModelESRNN(max_epochs=max_epochs, batch_size=batch_size, batch_size_test=batch_size_test, freq_of_test=freq_of_test,
               learning_rate=learning_rate, lr_scheduler_step_size=lr_scheduler_step_size, lr_decay=lr_decay,
               per_series_lr_multip=per_series_lr_multip, gradient_eps=gradient_eps, gradient_clipping_threshold=gradient_clipping_threshold,
               rnn_weight_decay=rnn_weight_decay, noise_std=noise_std,
               level_variability_penalty=level_variability_penalty,
               testing_percentile=testing_percentile, training_percentile=training_percentile, ensemble=ensemble,
               cell_type=cell_type,
               state_hsize=state_hsize, dilations= dilations,
               add_nl_layer=add_nl_layer, seasonality=seasonality, input_size=input_size, output_size=output_size,
               frequency=frequency, max_periods=max_periods, random_seed=random_seed, device=device)
    m.train(train_dc)
    y_predict = m.predict(validation_dc) 
    y_predict_df = y_predict.to_df()

    score = MP.MAPE(validation_df, y_predict_df)
    
    return score, (max_epochs, batch_size, input_size, output_size)


def validation_rolling(input_dc: DataCollection, num_split: int, numTest: int,
                    max_epochs=15, batch_size=1, batch_size_test=128, freq_of_test=-1,
                    learning_rate=1e-3, lr_scheduler_step_size=9, lr_decay=0.9,
                    per_series_lr_multip=1.0, gradient_eps=1e-8, gradient_clipping_threshold=20,
                    rnn_weight_decay=0, noise_std=0.001,
                    level_variability_penalty=80,
                    testing_percentile=50, training_percentile=50, ensemble=False,
                    cell_type='LSTM',
                    state_hsize=40, dilations=[[1, 2], [4, 8]],
                    add_nl_layer=False, seasonality=[4], input_size=4, output_size=8,
                    frequency=None, max_periods=20, random_seed=1):
    import time

    scores_list = []
    train_val_dic = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i in range(num_split):
        train, validation = input_dc.split(numTest = numTest)
        train_val_dic[i] = [train, validation]
        input_dc = train

    # record score of error
    total_score = 0
    elapse = 0
    for i in range(num_split-1, -1, -1):
        train_dc = train_val_dic[i][0]
        validation_dc = train_val_dic[i][1]

        validation_df = validation_dc.to_df()
        start_time = time.time()
        m = ModelESRNN(max_epochs=max_epochs, batch_size=batch_size, batch_size_test=batch_size_test, freq_of_test=freq_of_test,
               learning_rate=learning_rate, lr_scheduler_step_size=lr_scheduler_step_size, lr_decay=lr_decay,
               per_series_lr_multip=per_series_lr_multip, gradient_eps=gradient_eps, gradient_clipping_threshold=gradient_clipping_threshold,
               rnn_weight_decay=rnn_weight_decay, noise_std=noise_std,
               level_variability_penalty=level_variability_penalty,
               testing_percentile=testing_percentile, training_percentile=training_percentile, ensemble=ensemble,
               cell_type=cell_type,
               state_hsize=state_hsize, dilations= dilations,
               add_nl_layer=add_nl_layer, seasonality=seasonality, input_size=input_size, output_size=output_size,
               frequency=frequency, max_periods=max_periods, random_seed=random_seed, device=device)
        m.train(train_dc)
        y_predict = m.predict(validation_dc) 
        y_predict_df = y_predict.to_df()

        score = MP.MAPE(validation_df, y_predict_df)
        elapse += time.time() - start_time
        scores_list.append(score)
        total_score += score

    score = total_score / num_split

    return score, scores_list, elapse/num_split, (max_epochs, batch_size, input_size, output_size)

#######################################################
# parameters
num_split = 6
numTest = output_size = 30
input_size = 30
max_epochs = 15
batch_size = 64
learning_rate = 1e-2
lr_scheduler_step_size = 9
lr_decay = 0.9
noise_std = 0.001
level_variability_penalty = 80
state_hsize = 40
dilation = [[1]]
add_nl_layer = False
seasonality = [5]
# action
path = os.path.join('test','Data','Daily')
dic = preprocess.read_file(path)
series_list = []
for k, v in dic.items():
    df, cat = v
    df = preprocess.single_price(df, k)
    series_list.append(DataSeries(cat, 'daily', df))
collect = DataCollection('RollingValidation', series_list)
input_dc, _ = collect.split(numTest = 2 * numTest)

score, _ = validation_simple(input_dc, numTest=numTest, max_epochs=max_epochs, batch_size=batch_size, 
                learning_rate=learning_rate, lr_scheduler_step_size=lr_scheduler_step_size, lr_decay=lr_decay,
                noise_std=noise_std,level_variability_penalty=level_variability_penalty, 
                state_hsize=state_hsize, dilations=dilation,
                add_nl_layer=add_nl_layer, seasonality=seasonality, 
                input_size=input_size, output_size=output_size,frequency='D',random_seed=1)

print("--- score of this config in simple validation is %s ---" % score)
# print("---  time of this config is %s seconds ---" % (time.time() - start_time))

score, total_score, elapse, _ = validation_rolling(input_dc, num_split=num_split, numTest=numTest, max_epochs=max_epochs, batch_size=batch_size, 
                learning_rate=learning_rate, lr_scheduler_step_size=lr_scheduler_step_size, lr_decay=lr_decay,
                noise_std=noise_std,level_variability_penalty=level_variability_penalty, 
                state_hsize=state_hsize, dilations=dilation,
                add_nl_layer=add_nl_layer, seasonality=seasonality, 
                input_size=input_size, output_size=output_size,frequency='D',random_seed=1)

print("--- score of this config in rolling validation is %s ---" % round(score, 4))
print("--- individual score of this config in rolling validation are %s ---" % list(map(lambda x: round(x, 4), total_score)))
print("---  time of this config is %s seconds ---" % (round(elapse, 4)))
