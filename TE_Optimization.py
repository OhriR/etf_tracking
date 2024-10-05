
"""
Created on Thu Apr 4 03:19:55 2024

@author: Riya Ohri



"""
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

#pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() 
import datetime 

import warnings
warnings.filterwarnings("ignore")

# load module with utility functions, including optimization 
import risk_opt_2Student as riskopt 

    
def tracking_error(wts_active,cov):
    TE = np.sqrt(np.transpose(wts_active)@cov@wts_active)
    return TE

# function to get the price data from yahoo finance 
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  #return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date'])) # old
  # new - force it not to sort by the ticker alphabetically
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date'],sort=False))


# function to get the return data calculated from price data 
# retrived from yahoo finance 
def getReturns(tickers, start_dt, end_dt, freq='monthly'): 
    px_data = getDataBatch(tickers, start_dt, end_dt)
    # Isolate the `Adj Close` values and transform the DataFrame
    px = px_data[['Adj Close']].reset_index().pivot(index='Date', 
                           columns='Ticker', values='Adj Close')
    if (freq=='monthly'):
        px = px.resample('M').last()
        
    # Calculate the daily/monthly percentage change
    ret = px.pct_change().dropna()
    
    ret.columns = tickers
    return(ret)
    

    
if __name__ == "__main__":
    
    TickerNWeights = pd.read_excel('equity.xlsx', sheet_name='XLF', header=0, index_col=0)
    Ticker_AllStock_XLF = TickerNWeights['Symbol']
    wts_AllStock_XLF = 0.01*TickerNWeights['Weight']
    #Price_AllStock_XLF = TickerNWeights['Price']
    
    #%% get historical stock price data
    Flag_downloadData = False
    # define the time period 
    start_dt = datetime.datetime(2006, 1, 1)
    end_dt = datetime.datetime(2017, 12, 31)
    
    if Flag_downloadData:
        # Define the ticker symbol for XLF
        ticker_symbol = "XLF"
        
        # Define the start and end dates for the historical data
        start_date = "2006-01-01"
        end_date = "2017-12-31"
        
        # Download historical data for XLF
        XLF_Data = yf.download(ticker_symbol, start=start_date, end=end_date)

        #
        stock_data = getDataBatch(Ticker_AllStock_XLF, start_dt, end_dt)
        # Isolate the `Adj Close` values and transform the DataFrame
        Price_AllStock_XLF = stock_data.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
        Price_AllStock_XLF = Price_AllStock_XLF[list(Ticker_AllStock_XLF)]
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter('IndexPrice_XLF1.xlsx', engine='xlsxwriter')
        Price_AllStock_XLF.to_excel(writer, sheet_name='Price',startrow=0, startcol=0, header=True, index=True)
        XLF_Data.to_excel(writer, sheet_name='PriceXLF',startrow=0, startcol=0, header=True, index=True)
        writer.close()
    else:
        Price_AllStock_XLF = pd.read_excel('IndexPrice_XLF1.xlsx', sheet_name='Price',
                        header=0, index_col = 0)
        XLF_Data = pd.read_excel('IndexPrice_XLF1.xlsx', sheet_name='PriceXLF',
                        header=0, index_col = 0)
    
    #%%
    # Returns
    ret_AllStock = Price_AllStock_XLF.pct_change().dropna()
    ret_XLFIdx = XLF_Data['Adj Close'].pct_change().dropna().to_frame('Return')
    # Scale return data by a factor. It seems that the optimizer fails when the values are too close to 0
    scale = 1
    ret_AllStock = ret_AllStock*scale
    ret_XLFIdx = ret_XLFIdx*scale
    #
    num_periods, num_stock = ret_AllStock.shape
    
    #%%
    # Calulate Covariance Matrix
    #
    
    lamda = 0.94
    # vol of the assets 
    vols = ret_AllStock.std()
    rets_mean = ret_AllStock.mean()
    # demean the returns
    ret_AllStock = ret_AllStock - rets_mean
    
    # var_ewma calculation of the covraiance using the function from module risk_opt.py
    var_ewma = riskopt.ewma_cov(ret_AllStock, lamda)
    #var_ewma_annual = var_ewma*252 #Annualize
    # take only the covariance matrix for the last date, which is the forecast for next time period
    cov_end = var_ewma[-1,:]
    #
    cov_end_annual = cov_end*252 #Annualize
    std_end_annual = np.sqrt(np.diag(cov_end))*np.sqrt(252)
    # calculate the correlation matrix
    corr = ret_AllStock.corr()
    
    
    #%%
    # tracking error optimization
    #
    wts_active = np.zeros([num_stock,1])
    wts_active[0] = 0.05
    wts_active[-1] = -0.05
    #np.transpose(wts_active)@cov_end@wts_active
    TE1 = tracking_error(wts_active,cov_end)
    #
    # Test case - Full Replication : minize TE to zero should produce a fund with wts like those of the index
    #
    # define constraints
    b_ = [(0.0,1.0) for i in range(len(rets_mean))]  # no shorting 
    c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })   # Sum of active weights = 100%
    # calling the optimization function
    
    wts_min_trackingerror = riskopt.opt_min_te(wts_AllStock_XLF, cov_end_annual, b_, c_)
    # calc TE achieved
    wts_active1 = wts_min_trackingerror - wts_AllStock_XLF
    TE_optimized = tracking_error(wts_active1,cov_end)
    print('\nfull replication TE = {0:.5f} bps'.format(TE_optimized*10000))
    
    
    #
    # Test case - use only the top market cap stocks with highest index weights
    #
    num_topwtstock_2include = 15 #bad with 13, 16, 17, 18, 19, 20, 21,
    # only the top weight stocks + no shorting 
    b1a_ = [(0.0,1.0) for i in range(num_topwtstock_2include)]
    # exclude bottom weighted stocks
    b1b_ = [(0.0,0.0000001) for i in range(num_topwtstock_2include,num_stock)]
    b1_ = b1a_ + b1b_ # combining the constraints
    #b1_[num_topwtstock_2include:-1] = (0.0,0.0)
    c1_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })  # Sum of active weights = 100%
    # Calling the optimizer
    wts_min_trackingerror2 = riskopt.opt_min_te(wts_AllStock_XLF, cov_end_annual, b1_, c1_)
    # calc TE achieved
    wts_active2 = wts_min_trackingerror2 - wts_AllStock_XLF
    TE_optimized2 = tracking_error(wts_active2,cov_end_annual)
    print('{0} top weighted stock replication TE = {1:5.2f} bps'.format(num_topwtstock_2include, TE_optimized2*10000))
    
    # looping through number of stocks and save the history of TEs
    num_stock_b = 20
    num_stock_e = 55
    numstock_2use = range(num_stock_b,num_stock_e)
    wts_active_hist = np.zeros([len(numstock_2use), num_stock])
    TE_hist = np.zeros([len(numstock_2use), 1])
    count = 0
    
    for i in numstock_2use:
        # only the top weight stocks + no shorting 
        b1_c_a_ = [(0.0,1.0) for j in range(i)] 
        # exclude bottom weighted stocks
        b1_c_b_ = [(0.0,0.0000001) for j in range(i,num_stock)] 
        b1_curr_ = b1_c_a_ + b1_c_b_
        wts_min_curr = riskopt.opt_min_te(wts_AllStock_XLF, cov_end_annual, b1_curr_, c1_)
        wts_active_hist[count,:] = wts_min_curr.transpose()
        TE_optimized_c = tracking_error(wts_min_curr-wts_AllStock_XLF,cov_end_annual)
        TE_hist[count,:] = TE_optimized_c*10000# in bps
        count = count+1
        
        del b1_curr_, b1_c_a_, b1_c_b_,TE_optimized_c,wts_min_curr
    #
    #%%%
    #  Plot bars of weights
    #
    figure_count = 1
    # ---  create plot of weights fund vs benchmark
    plt.figure(figure_count)
    figure_count = figure_count+1
    fig, ax = plt.subplots(figsize=(18,10))
    index = np.arange(len(wts_AllStock_XLF))
    bar_width = 0.35
    opacity = 0.8
     
    rects1 = plt.bar(index, wts_AllStock_XLF, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Index Weight')
     
    rects2 = plt.bar(index + bar_width, wts_min_trackingerror2, bar_width,
                     alpha=opacity,
                     color='g',
                     label='ETF fund Weight')
     
    plt.xlabel('Ticker', fontsize=18)
    plt.ylabel('Weights', fontsize=18)
    plt.xticks(index + bar_width, (Ticker_AllStock_XLF), fontsize=12)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=18)
    plt.legend(fontsize=20)
     
    plt.tight_layout()
    plt.show()
    
    
    #------plot TE as a function of number of stocks -------------
    plt.figure(figure_count)
    figure_count = figure_count+1
    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(range(num_stock_b,num_stock_e), TE_hist, 'b')
    plt.xlabel('Number of stocks in ETF', fontsize=18)
    plt.ylabel('Optimized Tracking Error (bps)', fontsize=18)
    plt.title('XLF Industrial 73 ETF', fontsize=18)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
# %%%




import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

#pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() 
import datetime 

import warnings
warnings.filterwarnings("ignore")

# load module with utility functions, including optimization 
import risk_opt_2Student as riskopt 

    
def tracking_error(wts_active,cov):
    TE = np.sqrt(np.transpose(wts_active)@cov@wts_active)
    return TE

# function to get the price data from yahoo finance 
def getDataBatch(tickers, startdate, enddate):
  def getData(ticker):
    return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
  datas = map(getData, tickers)
  #return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date'])) # old
  # new - force it not to sort by the ticker alphabetically
  return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date'],sort=False))


# function to get the return data calculated from price data 
# retrived from yahoo finance 
def getReturns(tickers, start_dt, end_dt, freq='monthly'): 
    px_data = getDataBatch(tickers, start_dt, end_dt)
    # Isolate the `Adj Close` values and transform the DataFrame
    px = px_data[['Adj Close']].reset_index().pivot(index='Date', 
                           columns='Ticker', values='Adj Close')
    if (freq=='monthly'):
        px = px.resample('M').last()
        
    # Calculate the daily/monthly percentage change
    ret = px.pct_change().dropna()
    
    ret.columns = tickers
    return(ret)
    

    
if __name__ == "__main__":
    
    TickerNWeights = pd.read_excel('equity.xlsx', sheet_name='XLF', header=0, index_col=0)
    Ticker_AllStock_XLF = TickerNWeights['Symbol']
    wts_AllStock_XLF = 0.01*TickerNWeights['Weight']
    #Price_AllStock_XLF = TickerNWeights['Price']
    
    #%% get historical stock price data
    Flag_downloadData = False
    # define the time period 
    start_dt = datetime.datetime(2018, 1, 1)
    end_dt = datetime.datetime(2023, 12, 31)
    
    if Flag_downloadData:
        # Define the ticker symbol for XLF
        ticker_symbol = "XLF"
        
        # Define the start and end dates for the historical data
        start_date = "2018-01-01"
        end_date = "2023-12-31"
        
        # Download historical data for XLF
        XLF_Data_t = yf.download(ticker_symbol, start=start_date, end=end_date)

        #
        stock_data_t = getDataBatch(Ticker_AllStock_XLF, start_dt, end_dt)
        # Isolate the `Adj Close` values and transform the DataFrame
        Price_AllStock_XLF_t = stock_data_t.reset_index().pivot(index='Date', columns='Ticker', values='Adj Close')
        Price_AllStock_XLF_t = Price_AllStock_XLF_t[list(Ticker_AllStock_XLF)]
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter('IndexPrice_XLF2.xlsx', engine='xlsxwriter')
        Price_AllStock_XLF_t.to_excel(writer, sheet_name='Price',startrow=0, startcol=0, header=True, index=True)
        XLF_Data_t.to_excel(writer, sheet_name='PriceXLF',startrow=0, startcol=0, header=True, index=True)
        writer.close()
    else:
        Price_AllStock_XLF_t = pd.read_excel('IndexPrice_XLF2.xlsx', sheet_name='Price',
                        header=0, index_col = 0)
        XLF_Data_t = pd.read_excel('IndexPrice_XLF2.xlsx', sheet_name='PriceXLF',
                        header=0, index_col = 0)
    
    #%%
    # Returns
    ret_AllStock_t = Price_AllStock_XLF_t.pct_change().dropna()
    ret_XLFIdx_t = XLF_Data_t['Adj Close'].pct_change().dropna().to_frame('Return')
    # Scale return data by a factor. It seems that the optimizer fails when the values are too close to 0
    scale = 1
    ret_AllStock_t = ret_AllStock_t*scale
    ret_XLFIdx_t = ret_XLFIdx_t*scale
    #
    num_periods_t, num_stock_t = ret_AllStock.shape
    
    #%%
    # Calulate Covariance Matrix
    #
    
    lamda = 0.94
    # vol of the assets 
    vols_t = ret_AllStock_t.std()
    rets_mean_t = ret_AllStock_t.mean()
    # demean the returns
    ret_AllStock_t = ret_AllStock_t - rets_mean_t
    
    # var_ewma calculation of the covraiance using the function from module risk_opt.py
    var_ewma_t = riskopt.ewma_cov(ret_AllStock_t, lamda)
    #var_ewma_annual = var_ewma*252 #Annualize
    # take only the covariance matrix for the last date, which is the forecast for next time period
    cov_end_t = var_ewma_t[-1,:]
    #
    cov_end_annual_t = cov_end_t*252 #Annualize
    std_end_annual_t = np.sqrt(np.diag(cov_end_t))*np.sqrt(252)
    # calculate the correlation matrix
    corr_t = ret_AllStock_t.corr()
    

TE_forecast=tracking_error(wts_active2,cov_end_annual)    
TE_realized=tracking_error(wts_active2,cov_end_annual_t)    
    
print(TE_forecast)
print(TE_realized)
#%%%

import pandas as pd
import numpy as np

# Create an empty list to store the results
results = []

# Assuming ret_AllStock_t is your DataFrame containing the returns data
num_splits = 24
split_ret_AllStock_t = np.array_split(ret_AllStock_t, num_splits)

# Now you can access each split in a for loop
for i, df_split in enumerate(split_ret_AllStock_t):
    # Calculate the end date for the quarter
    
    end_date = df_split.index[-1]
    

    x = df_split
    lamda = 0.94
    # vol of the assets 
    vols_t = x.std()
    rets_mean_t = x.mean()
    # demean the returns
    x = x - rets_mean_t

    # var_ewma calculation of the covraiance using the function from module risk_opt.py
    var_ewma_t = riskopt.ewma_cov(x, lamda)
    # take only the covariance matrix for the last date, which is the forecast for next time period
    cov_end_t = var_ewma_t[-1,:]
    #
    cov_end_annual_t = cov_end_t*63 #Annualize

    TE_realized_quarterly = tracking_error(wts_active2, cov_end_annual_t)

    # Append the results to the list
    results.append({'End Date': end_date, 'TE Realized Quarterly': TE_realized_quarterly})

# Create a DataFrame from the list of results
result_df = pd.DataFrame(results)

# Print the resulting DataFrame
print(result_df)

    
import matplotlib.pyplot as plt

# Plot the forecasted and realized tracking error
plt.figure(figsize=(10, 6))
plt.plot(result_df['End Date'], result_df['TE Realized Quarterly'] * 10000, label='Realized TE (bps)', marker='o')
plt.axhline(y=TE_forecast * 10000, color='r', linestyle='--', label='Forecasted TE (bps)')
plt.xlabel('Date')
plt.ylabel('Tracking Error (bps)')
plt.title('Forecasted vs Realized Tracking Error')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
