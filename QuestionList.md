## Questions
1. Q: Historical portfolio & Enhanced portfolio, i.e, example on the email on 8/2/2020. Do we use forecast (2017-2019) with hitorical data(1990-2017) to calculate mean and cov or use only the forecast(2017-2019) to forecast? How much data is enough for estimating mean and cov matrix for portfolio construction? How many forcast do we need to generate in order to have a good covariance estimation? 

3. (last) Q: How to calculate covariance on rebalanced portfolio? reblanced frequency?
4. Q: Portfolio metrics calculation on three time periods: whole dataset? Zhilin asks
5. Q: Visualization?
6. (last) Q: How to do distribution forecast on asset price?

8. Q: How to train recommender for telescope? What data we use to feed in the training process?
9. Q: Seasonality for daily data? ESRNN use 7, we think for our case should be 252.
10. Q: We can only find daily & monthly data, how do we obtain data in other frequency?(i.e., yearly) If we want to calculate the yearly data, how do we do that?
----------

1. Three ways to intepret the results: the two ways CH mentioned; forecast the future time data point and use them to calculate Covariance matrix and mean, compare with traditional way of using historical data.
2. ESRNN & Telescope seems to be only able to forecast a short period of time (in M4, ESRNN forecasted 15 days of data), we think we could potenially forecast 30 days. But 30 days might not be enough to estimate covariance matrix, we might want to mix them with previous historical data. But mixing too much historical data might give very similar result with historical data. 
