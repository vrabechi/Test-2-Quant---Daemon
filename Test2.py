from datetime import date
import statistics
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from scipy.optimize import minimize
from scipy.stats import norm

#set portfolio assets (10 highest marketcap stocks in US)
assets=['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB', 'TSLA', 'BRK-A', 'VNDA', 'V', 'JPM']

#initialize dataframes
stock_prices=pd.DataFrame()
pf_returns=pd.DataFrame()

#collecting data for the stocks in the list
for a in assets:
    #collect price data for 4 months (1 for creating the initial portfolio + 3 for the holding period)
    stock_prices[a]=wb.DataReader(a, data_source = 'yahoo', start = '2021-5-1', end = '2021-8-31')['Adj Close']

#treating stock data to get returns and covariance matrix of returns
pf_returns=stock_prices.pct_change()

#funcion returns assets' risk weight based on portfolio weights and covariance matrix
def risk_allocation(w, cov_matrix):
    #treat variables as a matrix to use numpy operators
    cov=np.matrix(cov_matrix)
    w=np.matrix(w)

    #calculate portfolio variance
    portfolio_std=np.sqrt(w*(cov*w.T))

    #risk contribution of each asset
    asset_risk_allocation=np.multiply(w.T,(cov*w.T)/portfolio_std)

    #returns assets' risk weight
    return asset_risk_allocation/portfolio_std

#function to be minimized in order to have equal risk allocation
def cont_error(w, args):
    #extracting variables from args: target risk weight (equal weights), and covariance matrix
    target_w=args[0]
    cov_matrix=args[1]

    #calculating risk allocation considering the current iteration 
    risk_alloc=risk_allocation(w,cov_matrix)

    #treat variables as a matrix to use np operators
    target_w=np.matrix(target_w)

    #function to be minimized is the sum of the squared differences between desired and actual risk weights
    return sum(np.square(risk_alloc-target_w.T))[0,0]

#function to rebalance the portfolio once per month
def rebal(date, pf_returns_data, previous_days, target_risk_w, w_0):

    #select the data from the period of interest
    pf_returns_data=pf_returns_data.loc[:(initial_date)].iloc[-previous_days:]

    #calculate covariance matrix
    cov_matrix=pf_returns_data.cov()

    #find portfolio composition through iteration
    weights_calculation= minimize(cont_error, w_0, args=[target_risk_w, cov_matrix_0], method='SLSQP', constraints=constraints)

    #returns weights from the previous calculation
    return weights_calculation.x

#------------------------------------------------------------------------------------------------------------------

#assembling the portfolio on June 1st
#collecting data from the past 21 days to calculate initial composition
previous_days=21
initial_month=6
initial_day=1
initial_year=2021
initial_date=date(initial_year,initial_month,initial_day)

#get data from the previous 21 days
pf_returns_0=pf_returns.loc[:(initial_date)].iloc[-previous_days:]

#calculate covariance matrix of initial dataset
cov_matrix_0=pf_returns_0.cov()

#set the constraints and inputs for the later calculation of portfolio weights with the function "minimize"
#total weight must equal 1
constraints={'type': 'eq', 'fun': lambda x: sum(x)-1}
#the objective is equal risk weight
target_risk_w=[0.1] * 10
#initial guess of portfolio composition for iteration
w_0=[0.1] * 10


#use previously defined functions to calculate initial weights of portfolio
weights_calculation= minimize(cont_error, w_0, args=[target_risk_w, cov_matrix_0], method='SLSQP', constraints=constraints)
initial_weights=weights_calculation.x
#print initial portfolio composition
print('The initial portfolio composition is:')
i=0
for asset in assets:
    print(asset + ': ' +str(round(initial_weights[i]*100,2)) + '%')
    i+=1

#now we will go through the holding period day by day, calculating the updated... 
#portfolio composition, and rebalancing once per month
current_month=initial_month
end_date=date(2021,8,31)
current_weights=np.matrix(initial_weights)
portfolio_return=[]
buy_list=[]
sell_list=[]
cumulative_portfolio_return=1
cumulative_portfolio_return_plot=[]
rebalancing_count=1

for day in pf_returns.loc[initial_date:end_date].index:
#while current_date <= end_date:

    #return of individual stocks on current day
    current_returns=pf_returns.loc[day:day].to_numpy()[0,:]

    #return of portfolio
    portfolio_return.append(sum(np.dot(current_weights,current_returns.T))[0,0])
    cumulative_portfolio_return=cumulative_portfolio_return*(1+portfolio_return[-1])
    cumulative_portfolio_return_plot.append(cumulative_portfolio_return)
    
    #adjust weights for changes in stock prices and total portfolio return
    current_weights=np.multiply(current_weights,current_returns+1)/(1+portfolio_return[-1])

    #assuming rebalancing trades are made at the end of first business day of each month
    if day.month != current_month:

        #update initial guess
        w_0=current_weights

        print('\nNecessary trades for rebalancing after month ' + str(rebalancing_count) + " in terms of % of total portfolio:")

        #recalculating weights based on data of the last 21 days
        current_weights_rebal=rebal(day, pf_returns, previous_days, target_risk_w, w_0)

        #calculating difference between equal risk weights and current weights, in order to fnid the necessary trades for rebal 
        weight_diff=current_weights_rebal-current_weights

        #assigning each asset to buy or sell in accordance with the weight differentials 
        for asset in assets:
            diff=weight_diff[0,assets.index(asset)]
            if diff<=0: 
                sell_list.append([asset,round(diff*100,4)])
            else: 
                buy_list.append([asset,round(diff*100,4)])

        #print trade instructions for rebalancing
        for asset, diff in sell_list:
            print('Sell ' + str(-diff) + '% of the asset ' + asset)
        for asset,diff in buy_list:
            print('Buy ' + str(diff) + '% of the asset ' + asset)
        
        #clear lists for next rebalancing
        buy_list.clear()
        sell_list.clear()

        #update weights variable for post trades weights
        current_weights=np.matrix(current_weights_rebal)

        #update auxiliar variables
        rebalancing_count+=1
        current_month=day.month

#getting risk free rate from us treasury, using 3 months yield based on holding period of the portfolio
#https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yieldYear&year=2021
rf=0.02 #%

#calcularing portfolio's return, volatility and sharpe based on daily returns
cumulative_portfolio_return=round((cumulative_portfolio_return-1)*100,2)
portfolio_std=round(statistics.stdev(portfolio_return)*100,2)
sharpe=round((cumulative_portfolio_return-rf)/portfolio_std,2)
print('\nPortfolio total holding period return: ' + str(cumulative_portfolio_return) + '%')
print('Portfolio volatility: ' + str(portfolio_std) + '%')
print('Portfolio\'s Sharpe Index: ' + str(sharpe))

#calculating VaR for daily return, with a 5% probaility
VaR=round((statistics.mean(portfolio_return)*100-norm.ppf(0.95)*portfolio_std),2)
print('Value at Risk: There is a 5% probablity that the portfolio value will decline at least ' + str(-VaR) +'% on a given day')

