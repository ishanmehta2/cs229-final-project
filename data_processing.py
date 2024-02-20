import pandas as pd
import numpy as np

sf_df = pd.read_csv('weather_data/sf.csv')
ne_df = pd.read_csv('ne.csv')
pats_data = pd.read_csv('pats_data.csv')

ne_df['datetime'] = pd.to_datetime(ne_df['datetime']).dt.date
pats_data['game_date'] = pd.to_datetime(pats_data['game_date']).dt.date

ne_dates = set(ne_df['datetime'])
pats_dates = set(pats_data['game_date'])

# Find common dates
common_dates = ne_dates.intersection(pats_dates)

print("hello")
# Dictionary definitions

feature_dict = {'play_id', 'game_id', 'home_team', 'away_team' 'yardline_100', 'game_date', 'game_seconds_remaining',
                'down', 'ydstogo', 'play_type', 'no_huddle', 'shotgun', 'pass_length', 'pass_location', 'run_location',
                'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'score_differential', 'roof', 'surface', 'offense_formation',
                'offense_personnel', 'defenders_in_box', 'defense_personnel'}

new_pats_data = pats_data[['play_id', 'game_id', 'home_team', 'away_team', 'yardline_100', 'game_date', 'game_seconds_remaining',
                'down', 'ydstogo', 'play_type', 'no_huddle', 'shotgun', 'pass_length', 'pass_location', 'run_location',
                'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'score_differential', 'roof', 'surface', 'offense_formation',
                'offense_personnel', 'defenders_in_box', 'defense_personnel']]

# Format: {home team: weather.csv for home team}
team_to_weather_mapping = {'ARI': 'weather_data/phoenix.csv', 'ATL' : 'weather_data/atlanta.csv', 'BAL' : 'weather_data/baltimore.csv', 
                           'BUF' : 'weather_data/buffalo.csv', 'CAR' : 'weather_data/carolina.csv', #'CHI' : 'weather_data/chicago.csv',
                           'CIN' :  'weather_data/cincinnati.csv', 'CLE' : 'weather_data/cleveland.csv', 'DAL' : 'weather_data/dallas.csv',
                           'DEN' : 'weather_data/denver.csv', #'DET' : 'weather_data/detroit.csv', 
                           'GB' : 'weather_data/gb.csv', 'HOU' :'weather_data/houston.csv',
                           'IND' : 'weather_data/indianapolis.csv', 'JAX' :'weather_data/jax.csv', 'KC' : 'weather_data/kc.csv', 'LV' : 'weather_data/lv.csv',
                           'LAC' :  'weather_data/la.csv', 'LAR' : 'weather_data/la.csv', 'MIA' : 'weather_data/miami.csv', #'MIN' : 'weather_data/min.csv', 
                           'NE' : 'weather_data/ne.csv', 'NO' : 'weather_data/no.csv', 'NYG' :'weather_data/ny.csv', 'NYJ' : 'weather_data/ny.csv', 'PHI' :'weather_data/phi.csv',
                           'PIT' : 'weather_data/pit.csv', 'SF' : 'weather_data/sf.csv', 'SEA' : 'weather_data/sea.csv', 'TB' : 'weather_data/tb.csv', 'TEN' : 'weather_data/ten.csv', 'WAS' : 'weather_data/was.csv'}

new_mapping = {}
    
# Iterate through the original dictionary
for team, _ in team_to_weather_mapping.items():
    # For each team, create a new value in the specified format and add it to the new dictionary
    new_mapping[team] = f'{team.lower()}_data.csv'

# print(new_mapping)

team_data = {}
teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NY', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']
cur_teams = ['ARI', 'BUF', 'MIA', 'NE', 'NY', 'SEA', 'SF']

# Read in all the team data
for team in teams:
       # Create the variable name string
    filename = f'{team.lower()}_data.csv'  # Construct the file name based on the team name
    
    try:
        # Read the CSV file and store the DataFrame in the dictionary with the created variable name as key
        team_data[team] = pd.read_csv('updated_updated_team_data/' + filename)
    except FileNotFoundError:
        print(f'File not found for {team}, skipping.')

# print(team_data)

weather_mapping = {}


for team in teams:
      # Create the variable name string
    filename = f'{team.lower()}.csv'  # Construct the file name based on the team name
    print(filename)
    try:
        # Read the CSV file and store the DataFrame in the dictionary with the created variable name as key
        weather_mapping[team] = pd.read_csv('weather_data/' + filename)
    except FileNotFoundError:
        print(f'File not found for {team}, skipping.')


print(weather_mapping)
# print(team_data)

# Updating weather to only be what we want it to be

for team in cur_teams:
    # print(weather_mapping[team])
    print(team_data[team])


# print(team_data)
for team in team_data:
    game_loc = []
    prev_row = None
    for index, row in team_data.iterrows():
        loc = row['home_team']
        data = row['game_date']
# Print common dates

# print(sorted(common_dates))

# print(pats_data.head())
    






    