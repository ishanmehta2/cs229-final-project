import pandas as pd
import numpy as np

df = pd.read_csv('weather.csv')

for i in range(10):
    # Get the value in the 'datetime' column for the ith row
    if df.iloc[i]['icon'] == 'cloudy':
    
        datetime_value = df.iloc[i]['datetime']
    
        print(datetime_value)




