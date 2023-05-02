# Importing the required Python libraries
import numpy as np
import pandas as pd


# A function to generate a specific number of data points having a cosine distribution.
def cosine_distribution(amplitude, bias, nr_data_points, randomness_level):
  """
  Generates a cosine distribution with given amplitude, bias, required number of data points 
  and randomness_level. It returns the respective numpy array of generated values.
  """

  frequency = 0.5 # An arbitrary value for the frequency (however it doesn't significantly affect the generated values)  
  nr_cosine_periods = 24 / nr_data_points # Calculating the number of cosine periods needed so that the data points
  # having a cosine shape, represent 1 entire cosine period.

  # Generating 24 evenly spaced x values within the interval (0, number of cosine periods)  
  x = np.linspace(0, nr_cosine_periods * 2 * np.pi / frequency, 24) 

  # Calculating the respective y values using the cosine distribution formula
  y = amplitude * np.cos(frequency * x) + bias

  # Making sure that the user input for the 'randomness_level' is not negative.
  randomness_level = max(0, randomness_level)
  # Adding the noise effect
  y += np.random.normal(0, 0.1, 24) * randomness_level

  # Returning the np array of y
  return y


# A function to generate 24 data points following a normal distribution.
def generate_peak(peak_val, respective_h, randomness_level, peak_randomness):
  """
    Generates a normal distribution containing 24 data points, with given mean ('respective_h') and
    then transforms it so that it represents the daily demand, where 'peak_val' is its
    peak value, 'respective_h' is the respective peak's hour. 
    It returns the respective numpy array containing the daily demand values for each hour.  
  """

  mean = respective_h
  std = 0.2 * mean

  # Numpy array containing hours from 0 to 23.
  x = np.arange(24)
  
  # Gaussian distribution equation which produces probability values.
  # x = mean value, corresponds with the highest probability value
  y = 1/(std * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((x - mean) / std) ** 2)

  # Probability values are between 0 and 1, but we need them to represent demand values.
  # We also need to add some random noise to the generate values, based on the peak_randomness
  # value
  peak_noise_factor = np.random.uniform(0, 0.2 * peak_randomness)
  y *= ((peak_val + peak_noise_factor) / np.max(y))

  # Adding general random noise just like we did for the cosine distribution.
  y += np.random.normal(0, 0.1 * randomness_level, 24) 

  return y


# A function that generates the daily demand for car sharing service in a german city
def generate_daily_demand_updated(cos_first_val, cos_nr_points, first_peak_val, 
                                  first_peak_resp_h, second_peak_val, second_peak_resp_h, 
                                  randomness_level, peak_randomness):
  """
    Generates the daily demand for car sharing service in a german city. It uses the helper functions
    defined above and returns a numpy array containing the hourly demand values (24 values) for a day.
  """

  # Calculating the cosine amplitude and bias values based on the demand value at 0 hour, which usually represents
  # both the maximum cosine value and also the difference between min and max values of the cosine distribution. 
  amplitude = cos_first_val / 2    
  bias = cos_first_val / 2 + 10
  # Generating the demand for the first 'cos_nr_points' hours.          
  y1 = cosine_distribution(amplitude, bias, cos_nr_points, randomness_level)

  # Generating the normal distribution containing the first demand peak.
  y2 = generate_peak(first_peak_val, first_peak_resp_h, randomness_level, peak_randomness)

  # Generating the normal distribution containing the second demand peak.
  y3 = generate_peak(second_peak_val, second_peak_resp_h, randomness_level, peak_randomness) 

  # Definining 3 "window" lists containing weight values, which will multiply the daily demand represented by
  # the cosine and 2 normal distributions.
  if (cos_nr_points < 8):
    window_1 = np.array([1] * cos_nr_points + [0] * (24 - cos_nr_points))
  else:  
    window_1 = np.array([1, 0.9] + [1] * 6 + [0] * 16)
  window_2 = np.array([0] * cos_nr_points + [1] * (24 - cos_nr_points - 10) + [1.1, 1.4] + [0] * 8)
  window_3 = np.array([0] * 16 + [1] * 6 + [1, 0.8])
  
  # Multiplying each of the values that we got from each distribution with the respective weights.
  result_1 = y1 * window_1
  result_2 = y2 * window_2
  result_3 = y3 * window_3

  # Adding the values of 3 lists in order to get the entire daily demand.
  final_result = result_1 + result_2 + result_3

  return final_result


# A function that generates the demand for a certain number of days (365 for an entire year)
def generate_demand(nr_days, randomness_level, peak_randomness):
  """
    Generates the demand for car sharing service in a german city for a given number
    of days, general randomness level and demand's peak specific randomness. It uses the helper 
    functions defined above and returns a Pandas DataFrame, containing 4 features 
    'Demand', 'Hour', 'Day of week' and 'Specific hour' (the same as 'Hour' column but it
    contains only values from 0 to 23, which are helpful while visualising the data).
  """

  weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

  # A numpy array which will contain the hourly demand values.
  demand = np.empty((0,))
  
  # Defining a list of dictionaries for the demand parameters of each day of the week. 
  # The values are based on the research paper that I should imitate.
  days = [
          {
            "cos_first_val": 90, 
            "cos_nr_points": 8, 
            "first_peak_val": 200, 
            "first_peak_resp_h": 12, 
            "first_peak_diff_max_min": 110, 
            "second_peak_val": 280, 
            "second_peak_resp_h": 18, 
            "second_peak_diff_max_min": 150
           },
          {
            "cos_first_val": 95, 
            "cos_nr_points": 8, 
            "first_peak_val": 180, 
            "first_peak_resp_h": 12, 
            "first_peak_diff_max_min": 80, 
            "second_peak_val": 240, 
            "second_peak_resp_h": 17,   
            "second_peak_diff_max_min": 140
           },    
          {
            "cos_first_val": 90, 
            "cos_nr_points": 8, 
            "first_peak_val": 150, 
            "first_peak_resp_h": 11, 
            "first_peak_diff_max_min": 50, 
            "second_peak_val": 260, 
            "second_peak_resp_h": 18, 
            "second_peak_diff_max_min": 190
           },    
          {
            "cos_first_val": 70, 
            "cos_nr_points": 6, 
            "first_peak_val": 240, 
            "first_peak_resp_h": 11.5, 
            "first_peak_diff_max_min": 90, 
            "second_peak_val": 300, 
            "second_peak_resp_h": 17, 
            "second_peak_diff_max_min": 210
           },    
          { 
            "cos_first_val": 50, 
            "cos_nr_points": 6, 
            "first_peak_val": 200, 
            "first_peak_resp_h": 11, 
            "first_peak_diff_max_min": 90, 
            "second_peak_val": 340, 
            "second_peak_resp_h": 16, 
            "second_peak_diff_max_min": 270
           },    
          {
            "cos_first_val": 50, 
            "cos_nr_points": 6, 
            "first_peak_val": 240, 
            "first_peak_resp_h": 7, 
            "first_peak_diff_max_min": 100, 
            "second_peak_val": 340, 
            "second_peak_resp_h": 12,  # increase the std value or just change the values here for the moment
            "second_peak_diff_max_min": 200
           },    
          {
            "cos_first_val": 60, 
            "cos_nr_points": 6, 
            "first_peak_val": 260, 
            "first_peak_resp_h": 12, 
            "first_peak_diff_max_min": 110, 
            "second_peak_val": 410, 
            "second_peak_resp_h": 19, 
            "second_peak_diff_max_min": 310
           }
          ]

  changing_factor = 1

  # Iterating for each day
  for i in np.arange(nr_days):
    day_index = i % 7
    day = days[day_index]
    cos_first_val = day["cos_first_val"]
    cos_nr_points = day["cos_nr_points"]
    first_peak_val = day["first_peak_val"]
    first_peak_resp_h = day["first_peak_resp_h"]
    first_peak_diff_max_min = day["first_peak_diff_max_min"]
    second_peak_val = day["second_peak_val"]
    second_peak_resp_h = day["second_peak_resp_h"]
    second_peak_diff_max_min = day["second_peak_diff_max_min"]

    # Storing the function result into 'daily_demand'
    daily_demand = generate_daily_demand_updated(cos_first_val, cos_nr_points, first_peak_val, 
                                  first_peak_resp_h,
                                  second_peak_val, second_peak_resp_h, 
                                  randomness_level,
                                  peak_randomness) * changing_factor
    
    # Extending the current 'demand' numpy array with each 'daily_demand'
    demand = np.concatenate((demand, daily_demand), axis=0)

    # Increasing/decreasing the demand every week, starting from the second one, using a random factor 
    if i % 7 == 6:
      increase_decrease_demand = np.random.choice(['increase', 'decrease', 'no change'], p=[0.3, 0.2, 0.5])
      if increase_decrease_demand == 'increase':
        changing_factor = np.random.choice([1.05, 1.1, 1.15])
      elif increase_decrease_demand == 'decrease':
        changing_factor = np.random.choice([0.85, 0.9, 0.95])
      else:
        changing_factor = 1

  # Rounding the demand values to the nearest integer and converting all the negative values to 0
  demand = np.round(demand)
  demand = np.clip(demand, 0, None)

  # Creating a Pandas DataFrame containing 2 columns, "Hour" and "Demand" 
  df = pd.DataFrame({'Demand' : demand,
                     'Hour' : np.arange(nr_days * 24)})
  
  # Adding 2 new columns for the respective day of week and the specific hour (0-23)
  df['Day of week'] = pd.Series(weekdays)[(df['Hour'] // 24) % 7].values
  df['Specific hour'] = df['Hour'] % 24

  return df