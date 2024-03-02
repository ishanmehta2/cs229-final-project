import numpy as np
import pandas as pd
import nfl_data_py as nfl

seasons = list(range(2016, 2023))

team_city_mapping = {'ARI': 'phoenix', 'ATL' : 'atlanta', 'BAL' : 'baltimore',
                    'BUF' : 'buffalo', 'CAR' : 'charlotte', 'CHI' : 'chicago',
                    'CIN' :  'cincinnati', 'CLE' : 'cleveland', 'DAL' : 'dallas',
                    'DEN' : 'denver', 'DET' : 'detroit',
                    'GB' : 'green bay', 'HOU' :'houston',
                    'IND' : 'indianapolis', 'JAX' :'jacksonville', 'KC' : 'kansas city', 'LV' : 'las vegas',
                    'LAC' :  'los angeles', 'LA' : 'los angeles', 'MIA' : 'miami', 'MIN' : 'minneapolis',
                    'NE' : 'new england', 'NO' : 'new orleans', 'NYG' :'east rutherford', 'NYJ' : 'east rutherford', 'PHI' :'philadelphia',
                    'PIT' : 'pittsburgh', 'SF' : 'san francisco', 'SEA' : 'seattle', 'TB' : 'tampa bay', 'TEN' : 'nashville', 'WAS' : 'washington dc'}

weather_data = pd.read_csv('/Users/ohmpatel/Documents/GitHub/cs229-final-project/weather_full.csv')

def process_team_data(seasons, weather_data):
    pbp_data = pd.DataFrame()

    season_data = nfl.import_pbp_data(years=seasons)
    pbp_data = pd.concat([pbp_data, season_data], axis=0)

    # remove the non pass/run plays
    offensive_snaps = pbp_data[(pbp_data['play_type'] == 'run') | (pbp_data['play_type'] == 'pass')]
    # sequence the plays by order in a game
    play_sequence_order = []
    curr_seq_num = 1
    prev_row = None
    for index, row in offensive_snaps.iterrows():
        if curr_seq_num == 1:
            play_sequence_order.append(curr_seq_num)
            curr_seq_num += 1
            prev_row = row
        else:
        #  print(prev_row['game_id'], row['game_id'])
            if prev_row['game_id'] == row['game_id']:
                curr_seq_num += 1
                play_sequence_order.append(curr_seq_num)
                prev_row = row
            else: # new game
                curr_seq_num = 1
                play_sequence_order.append(curr_seq_num)
    offensive_snaps['play_sequence_num'] = play_sequence_order

    offensive_snaps = offensive_snaps[['play_sequence_num', 'game_id', 'home_team', 'away_team', 'yardline_100', 'game_date', 'game_seconds_remaining',
                'down', 'ydstogo', 'play_type', 'no_huddle', 'shotgun', 'pass_length', 'pass_location', 'run_location',
                'posteam_timeouts_remaining', 'defteam_timeouts_remaining', 'score_differential', 'roof', 'surface', 'offense_formation',
                'offense_personnel', 'defenders_in_box', 'defense_personnel']]

    print('onto weather. we have ' + str(offensive_snaps.shape) + 'data to get through.')

    feelslike = []
    humidity = []
    conditions = []
    windspeed = []
    prev_row = None
    print('starting weather')
    for idx, row in offensive_snaps.iterrows():
        if not (idx % 5000):
            print(idx)
        # we have the start of a new game, need to add new weather query
        if prev_row is None or prev_row['home_team'] == row['home_team']:
            home_city = team_city_mapping[row['home_team']]
            date = row['game_date']
            matching_weather_row = weather_data.query('name == @home_city & datetime == @date')
            feelslike.append(list(matching_weather_row['feelslike'])[0])
            humidity.append(list(matching_weather_row['humidity'])[0])
            conditions.append(list(matching_weather_row['conditions'])[0])
            windspeed.append(list(matching_weather_row['windspeed'])[0])
        else:
            feelslike.append(feelslike[-1])
            humidity.append(humidity[-1])
            conditions.append(conditions[-1])
            windspeed.append(windspeed[-1])
    offensive_snaps['feelslike'] = feelslike
    offensive_snaps['humidity'] = humidity
    offensive_snaps['conditions'] = conditions
    offensive_snaps['windspeed'] = windspeed


    return offensive_snaps


data = process_team_data(seasons, weather_data)

saved = data.to_csv('/Users/ohmpatel/Documents/GitHub/cs229-final-project/big_data.csv')