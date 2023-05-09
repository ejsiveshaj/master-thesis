# Importing the required Python libraries
import numpy as np
import pandas as pd
import plotly.express as px


# Defining a function which imports a csv file as a Pandas DataFrame and returns only the demand column
def get_demand_column(file_path):
  '''It imports the csv file representing the simulated data, as a Pandas DataFrame and returns
  the demand column as a numpy array.'''

  # Importing the csv file containing the data for training the model
  train_df = pd.read_csv(file_path)

  # Returning the demand column as a numpy array
  return np.array(train_df["Demand"])


# Defining a function which converts the above datasets into the required format for the LSTM model
def create_dataset(data, look_back=1):
  '''Creates and returns the inputs and outputs for the LSTM model, for a given dataset and a look
  back range (number of timesteps).
  '''
  x, y = [], []

  for i in range(len(data) - look_back):
    x.append(data[i : (i + look_back)])
    y.append(data[i + look_back])

  x = np.array(x)
  y = np.array(y)

  # Reshaping x from [samples, look_back] into [samples, look_back, n_features]
  n_features = 1
  x = x.reshape((x.shape[0], x.shape[1], n_features))
    
  return x, y


# Defining a function which returns the demand scaling factor
def return_scaler(file_path):
  '''Imports the csv file as a Pandas DataFrame and 
  returns the demand scaling factor.'''

  # Importing the csv file containing the data for training the model
  train_df = pd.read_csv(file_path)

  return 1 / (4 * train_df['Demand'].max())


# Defining a function which plots the demand real values vs the predicted ones
def real_values_vs_predictions(df_file_path, predictions, look_back):
  '''Plots the demand real values vs the predicted ones,
  for a given csv file(train or test data), a numpy array containing
  the predictions made by the lstm model and a specific look back range.'''

  # Importing the csv file containing real values, as a Pandas DataFrame.
  real_val_df = pd.read_csv(df_file_path)

  # Creating a deep copy of the above DataFrame
  predictions_df = real_val_df.copy()

  # Updating the above DataFrame with the predicted values
  predictions_df.iloc[look_back:, 0] = predictions

  # Rounding the demand values to the nearest integer
  predictions_df['Demand'] = predictions_df['Demand'].round()

  # Concatenating both DataFrames and adding another column determining
  # the respective DataFrame for each data point
  df = pd.concat([real_val_df.assign(Dataset='Real value'), predictions_df.assign(Dataset='Predicted value')])

  # Plotting real demand values vs the predicted ones, using 2 different colors.
  fig = px.scatter(df, x='Hour', y='Demand', color='Dataset', hover_data = ['Specific hour', 'Day of week'])
  fig.show() 
