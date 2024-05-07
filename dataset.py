import yfinance
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

symbol="BTC-USD"
start="2014-01-01"
end="2024-01-01"
BUY=1
SELL=0
INCREASED=1
DECREASED=0
INDEX_OF_RETURNS_COLUMN=8
INDEX_OF_BUY_SELL_COLUMN_ADJ_CLOSE=6


def prepare_dataset():

    dataframe = yfinance.download(symbol,start,end)

    dataframe.info()
    dataframe.to_csv("btc-usd-2014-2024.csv")
    dataframe["Buy or Sell (Open)"] = numpy.where(dataframe["Open"].shift(-1)>dataframe["Open"],BUY,SELL)
    dataframe["Returns"]=dataframe["Adj Close"].pct_change()


    dataframe["Volume Increase or Decrease"] = numpy.where(dataframe["Volume"].shift(-1)>dataframe["Volume"],INCREASED,DECREASED)
    dataframe  = dataframe.dropna()


    y=dataframe.iloc[:,INDEX_OF_RETURNS_COLUMN].values
    x=dataframe.iloc[:,INDEX_OF_BUY_SELL_COLUMN_ADJ_CLOSE:INDEX_OF_RETURNS_COLUMN].values

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

    # return X_train, X_test, Y_train, Y_test
    
    return x, y
