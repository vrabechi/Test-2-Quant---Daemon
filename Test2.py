import pandas as pd
import numpy as np
import matplotlib.pyplot
from pandas_datareader import data as wb

assets=['AAPL', 'MSFT']
#'GOOG', 'AMZN', 'FB', 'TSLA', 'BRK-A', 'VNDA', 'V', 'JPM']
pf_data=pd.DataFrame()

for a in assets:
    #collect data for 4 months (1 for creating the initial portfolio + 3 for the holding period)
    pf_data[a]=wb.DataReader(a, data_source = 'yahoo', start = '2021-5-1', end = '2021-8-31')['Adj Close']


print(pf_data)

