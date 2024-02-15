import pandas as pd
import numpy as np

df = pd.read_csv('weather.csv')



while True:
    df.sort_values(by='datetime', inplace=True)

# Select every seventh day starting with the third day
# Since we start with the third day (index 2) and want every seventh day, we use slicing with a step of 7
    selected_days = df.iloc[2::7]
    break

print(selected_days)







# for i in range(10):
#     # Get the value in the 'datetime' column for the ith row
#     if df.iloc[i]['icon'] == 'cloudy':
    
#         datetime_value = df.iloc[i]['datetime']
    
#         print(datetime_value)




