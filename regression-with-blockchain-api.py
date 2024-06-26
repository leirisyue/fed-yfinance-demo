import os
import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

url = 'D:/test/fed_ml_flower_demo-master'
def read_files():
  files = []
  for _,_, file in os.walk("."):
    for file_name in file:
      if file_name.endswith("csv"):
        files.append(file_name)
  return files

def get_data(files):
  dataframe = None

  for file in files:
    if dataframe is None:
      dataframe = pandas.read_csv(f"{url}/data/{file}", names=["time",file.replace(".csv","")])
    else:
      temporary_dataframe = pandas.read_csv(f"{url}/data/{file}", names=["time",file.replace(".csv","")])
      dataframe = pandas.merge(dataframe,temporary_dataframe)
      
  dataframe = dataframe[1:]
  dataframe = dataframe.apply(pandas.to_numeric,errors="coerce")
  return dataframe

def draw_plot(dataframe):
  fig = plt.figure(figsize=(35,20))
  COLUMNS = 2
  ROWS =4
  columns = list(dataframe.columns)
  columns.remove("time")
  
  for index in range(len(columns)):
    fig.add_subplot(ROWS,COLUMNS,index+1)
    current_feature = columns[index]
    plt.plot(dataframe["time"],dataframe[current_feature])

  plt.show()
  
def get_XY(dataframe):
  NUMBER_OF_TRANSACTIONS_INDEX = 8
  MARKET_PRICE_INDEX = 7
  END_RANGE = NUMBER_OF_TRANSACTIONS_INDEX + 1
  
  X = dataframe.iloc[:, NUMBER_OF_TRANSACTIONS_INDEX:END_RANGE]
  Y = dataframe.iloc[:, MARKET_PRICE_INDEX]
  return X,Y

def draw_graph_Linear(X,Y):
  model = LinearRegression()
  model.fit(X,Y)
  plt.scatter(X,Y)
  plt.ylabel("Bitcoin Price")
  plt.xlabel("Feature: Number of Transactions")
  plt.plot(X, model.predict(X), color = "purple")
  
  
def draw_graph_Regression(X,Y):
  # X,Y = get_XY(dataframe)
  # fig = plt.figure(figsize=(35,20))
  # COLUMNS = 2
  # ROWS =4
  # columns = list(X.columns)
  
  # for index in range(len(columns)):
  #   fig.add_subplot(ROWS,COLUMNS,index+1)
  #   current_feature = columns[index]
  #   plt.plot(X[current_feature],Y)

  X_DEGREE = 2
  polynomial_features = PolynomialFeatures(degree = X_DEGREE)
  X_polynomial = polynomial_features.fit_transform(X)
  polynomial_features.fit(X_polynomial,Y)
  linear_model = LinearRegression()
  linear_model.fit(X_polynomial,Y)
  plt.scatter(X,Y)
  plt.plot(X,linear_model.predict(polynomial_features.fit_transform(X)),color="purple")
  plt.show()
  
def prepare_dataset():
  NUMBER_OF_TRANSACTIONS_INDEX = 8
  END_RANGE = NUMBER_OF_TRANSACTIONS_INDEX + 1
  MARKET_PRICE_INDEX = 7
  files = read_files()
  dataframe_reg = get_data(files)
  X=dataframe_reg.iloc[:, NUMBER_OF_TRANSACTIONS_INDEX:END_RANGE]
  Y=dataframe_reg.iloc[:, MARKET_PRICE_INDEX]
  return X,Y

if __name__ == "__main__":
  # data = pandas.read_csv(url + "/hash-rate.csv", engine="python", sep=',', encoding='latin-1')
  files = read_files()
  dataframe_reg = get_data(files)
  draw_plot(dataframe_reg)
  X,Y = get_XY(dataframe_reg)
  draw_graph_Linear(X,Y)
  draw_graph_Regression(X,Y)
  # print(dataframe_reg)
  # x, y = prepare_dataset()
  # val_split = int(0.2 * x.shape[0])
  # train_split = (x.shape[0] - val_split) // 2
  # print(train_split)