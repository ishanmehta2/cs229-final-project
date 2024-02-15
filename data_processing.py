import pandas as pd
import numpy as np

sf_df = pd.read_csv('SF_weather.csv')
ne_df = pd.read_csv('ne.csv')
pats_data = pd.read_csv('pats_data.csv')

ne_df['datetime'] = pd.to_datetime(ne_df['datetime']).dt.date
pats_data['game_date'] = pd.to_datetime(pats_data['game_date']).dt.date

ne_dates = set(ne_df['datetime'])
pats_dates = set(pats_data['game_date'])

# Find common dates
common_dates = ne_dates.intersection(pats_dates)

# Print common dates

print(sorted(common_dates))
    






    