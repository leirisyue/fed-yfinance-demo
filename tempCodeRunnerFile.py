
  dataframe = get_data(files)
  draw_plot(dataframe)
  X,Y = get_XY(dataframe)
  draw_graph_Linear(X,Y)
  draw_graph_Regression(X,Y)