import os
import torch
import pandas as pd
import numpy as np
from utils.Data import DataCollection, DataSeries
from Models.ModelESRNN import ModelESRNN
import utils.DataPreprocessing as preprocess
import utils.Validation as Vn


path = os.path.join('test','Data','Daily')
dic = preprocess.read_file(path)
series_list = []
for k, v in dic.items():
    df, cat = v
    df = preprocess.single_price(df, k)
    series_list.append(DataSeries(cat, 'daily', df))
collect = DataCollection('Simple', series_list)



dilation_list = [[[1]],[[5]],[[1,5]],[[1,3,5]],[[1,5,10]],[[1,5,20]],[[1,3,5,10]]
,[[1,3,5,20]],[[1,5,10,20]],[[1],[5]],
[[1],[3,5]],
[[1],[5,10]],
[[1],[5,20]],
[[1],[3,5,10]],
[[1],[3,5,20]],
[[1],[5,10,20]],
[[1,3],[5]],
[[1,5],[10]],
[[1,5],[20]],
[[1,3],[5,10]],[[1,3,5],[10]],
[[1,3],[5,20]],[[1,3,5],[20]],
[[1,5],[10,20]],[[1,5,10],[20]]
]

dil_one = [[[1,5]]]
# train/test split
input_dc, test_dc = collect.split(numTest = 2 * 90)

score_list = []
time_list = []
for dn in dil_one:
    import time
    start_time = time.time()
    score, _ = Vn.validation_simple(input_dc, numTest=90, max_epochs=15, batch_size=64, 
                    learning_rate=1e-3, lr_scheduler_step_size=9, lr_decay=0.9,
                    noise_std=0.001,level_variability_penalty=80, 
                    state_hsize=40, dilations=dn,
                    add_nl_layer=False, seasonality=[5], 
                    input_size=5, output_size=90,frequency='D',random_seed=1
                    )
    print("--- dilation :", dn," ---")
    print("--- score of this config is %s ---" % score)
    score_list.append(score)
    print("---  time of this config is %s seconds ---" % (time.time() - start_time))
    time_list.append(time.time() - start_time)

data = [dilation_list,score_list,time_list]
df = pd.DataFrame(data, index = ['dilation','score','time']).T
print(df)

