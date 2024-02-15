import pandas as pd
import numpy as np

df = pd.read_csv('sf.csv')

<<<<<<< Updated upstream
        datetime_value = df.iloc[i]['datetime']
#         datetime_value = df.iloc[i]['datetime']
    
        print(datetime_value)
#         print(datetime_value)
=======
while True:
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Sort the DataFrame by the 'datetime' column to ensure it's in chronological order
    df.sort_values(by='datetime', inplace=True)
>>>>>>> Stashed changes

    # Select every seventh day starting with the third day
    # Since we start with the third day (index 2) and want every seventh day, we use slicing with a step of 7
    selected_days = df.iloc[2::7]
    print(selected_days['datetime'])
    break

