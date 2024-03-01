import pandas as pd
import numpy as np


# Initial Run

# sf_df = pd.read_csv('weather_data/sf.csv')
# ne_df = pd.read_csv('ne.csv')
# pats_data = pd.read_csv('pats_data.csv')

# ne_df['datetime'] = pd.to_datetime(ne_df['datetime']).dt.date
# pats_data['game_date'] = pd.to_datetime(pats_data['game_date']).dt.date

# ne_dates = set(ne_df['datetime'])
# pats_dates = set(pats_data['game_date'])

# # Find common dates
# common_dates = ne_dates.intersection(pats_dates)

print("hello")
# Dictionary definitions

feature_dict = {'play_id', 'game_id', 'home_team', 'away_team' 'yardline_100', 'game_date', 'game_seconds_remaining',
                'down', 'ydstogo', 'play_type', 'no_huddle', 'shotgun', 'pass_length', 'pass_location', 'run_location',
                'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'score_differential', 'roof', 'surface', 'offense_formation',
                'offense_personnel', 'defenders_in_box', 'defense_personnel'}

# new_pats_data = pats_data[['play_id', 'game_id', 'home_team', 'away_team', 'yardline_100', 'game_date', 'game_seconds_remaining',
#                 'down', 'ydstogo', 'play_type', 'no_huddle', 'shotgun', 'pass_length', 'pass_location', 'run_location',
#                 'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'score_differential', 'roof', 'surface', 'offense_formation',
#                 'offense_personnel', 'defenders_in_box', 'defense_personnel']]

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

weather_mapping = {'ARI': 'weather_data/ari.csv', 'ATL' : 'weather_data/atl.csv', 'BAL' : 'weather_data/bal.csv', 
                           'BUF' : 'weather_data/buf.csv', 'CAR' : 'weather_data/car.csv', 'CHI' : 'weather_data/chi.csv',
                           'CIN' :  'weather_data/cin.csv', 'CLE' : 'weather_data/cle.csv', 'DAL' : 'weather_data/dal.csv',
                           'DEN' : 'weather_data/den.csv', 'DET' : 'weather_data/det.csv', 
                           'GB' : 'weather_data/gb.csv', 'HOU' :'weather_data/hou.csv',
                           'IND' : 'weather_data/ind.csv', 'JAX' :'weather_data/jax.csv', 'KC' : 'weather_data/kc.csv', 'LV' : 'weather_data/lv.csv',
                           'LAC' :  'weather_data/la.csv', 'LA' : 'weather_data/la.csv', 'MIA' : 'weather_data/mia.csv', 'MIN' : 'weather_data/min.csv', 
                           'NE' : 'weather_data/ne.csv', 'NO' : 'weather_data/no.csv', 'NYG' :'weather_data/ny.csv', 'NYJ' : 'weather_data/ny.csv', 'PHI' :'weather_data/phi.csv',
                           'PIT' : 'weather_data/pit.csv', 'SF' : 'weather_data/sf.csv', 'SEA' : 'weather_data/sea.csv', 'TB' : 'weather_data/tb.csv', 'TEN' : 'weather_data/ten.csv', 'WAS' : 'weather_data/was.csv'}


# for team in teams:
#       # Create the variable name string
#     filename = f'{team.lower()}.csv'  # Construct the file name based on the team name
#     print(filename)
#     try:
#         # Read the CSV file and store the DataFrame in the dictionary with the created variable name as key
#         weather_mapping[team] = pd.read_csv('weather_data/' + filename)
#     except FileNotFoundError:
#         print(f'File not found for {team}, skipping.')

for team in teams:
    try:
        temp = weather_mapping[team]
        new_temp = pd.read_csv('/Users/ishan/Desktop/cs229-final-project/' + temp)
        weather_mapping[team] = new_temp
        print("processsed")
        
    except:
        print('File not found for {team}, skipping.')


for team, df in team_data.items():
    # Define the file name for each team's CSV
    file_name = f"{team}.csv"
    
    # Save the DataFrame to CSV
    df.to_csv(file_name, index=False)  # Set index=False if you don't want the index in the CSV

print("CSV files have been saved for each team.")


for column in weather_mapping['ARI'].columns:
    team_data['ARI'][column] = None
    # print(team_data['ARI'].columns)

for team in cur_teams:
    team_vec = []
    recent = 0
    team_pd_df = team_data[team]
    
    team_weather_pd_df = weather_mapping[team]
    count = 0

    # Finding the home team 
    for i in range(1, len(team_pd_df)):
        try:
            home_team = str(team_pd_df.loc[i, 'home_team'])
            cur_date = team_pd_df.loc[i, 'game_date']
            if home_team in teams:
                weather_pd_df = weather_mapping[home_team]
                matching_row = weather_pd_df[weather_pd_df['datetime'] == cur_date]
                for col in matching_row.columns:
                    if pd.isnull(team_pd_df.loc[i, col]):
                        team_pd_df.loc[i, col] = matching_row[col].values[0]
    
        except Exception as e:
            print(f"Error processing row {i}: {e}")
        # except:
        #     team_pd_df = team_pd_df.drop(i)
        #     count += 1
        
        # print(team_pd_df.loc[i])
    print(len(team_pd_df), count)
        
        
    
        # Finding the weather from the home team city





    # vectorized_team_dates= set()
    # # if team == 'ARI':
    # for i in range(1, len(team_pd_df)):
        
    #         vectorized_team_dates.add(team_pd_df['game_date'][i])

    # for ind in team_pd_df:
    #     # for index in team_weather_pd_df:
    #     for i in range(recent + 1, len(team_weather_pd_df)):
    #         if team_weather_pd_df['datetime'][i] in vectorized_team_dates: 
    #             team_vec.append(i)
    #             recent = i
    
    # cut_down_df = pd.DataFrame(columns=['name', 'datetime', 'tempmax', 'tempmin', 'temp', 'feelslikemax',
    #    'feelslikemin', 'feelslike', 'dew', 'humidity', 'precip', 'precipprob',
    #    'precipcover', 'preciptype', 'snow', 'snowdepth', 'windgust',
    #    'windspeed', 'winddir', 'sealevelpressure', 'cloudcover', 'visibility',
    #    'solarradiation', 'solarenergy', 'uvindex', 'severerisk', 'sunrise',
    #    'sunset', 'moonphase', 'conditions', 'description', 'icon', 'stations'])
    
    # weather_df_sorted = team_weather_pd_df.sort_values(by='datetime', ascending=True)
    # team_data_sorted = team_pd_df.sort_values(by='game_date', ascending=True)
    # # weather_df_sorted = weather_df_sorted.rename(columns={'datetime': 'game_date'})

    
    # for ind in team_vec:
    # # Assuming 'ind' is a valid index in both DataFrames and they are aligned
    # # Extract row data from weather DataFrame
    #     row_data = weather_df_sorted.loc[ind]
        
    #     # Update the team_data_sorted DataFrame's corresponding row
    #     # Here, we iterate over each column that needs to be updated
    #     for col in weather_df_sorted.columns:
    #         # Assuming the same column names or you have a mapping of column names
    #         # Update the value in team_data_sorted at the same index and column
    #         if col in team_data_sorted.columns:
    #             team_data_sorted.at[ind, col] = row_data[col]
    
    # if team == 'ARI':
    #     specific_date = '2016-09-11'
    #     specific_row = team_data_sorted.query(f"game_date == '{specific_date}'")

    #     # Print each column value for the row
    #     if not specific_row.empty:
    #         print(f"Values for {specific_date}:")
    #         for column in specific_row.columns:
    #             print(f"{column}: {specific_row.iloc[0][column]}")
    #     else:
    #         print(f"No data found for {specific_date}")



            
        


    



    
            

    


    # print(team_vec)
    

















    