import numpy as np
import pandas as pd
import nfl_data_py as nfl

# teams
teams = ['NE', 'BUF', 'MIA', 'NYJ', 'SF', 'SEA', 'ARI', 'LA']

# Define the range of seasons
seasons = list(range(2016, 2023))

def process_team_data(team_abbrev):
  pbp_data = pd.DataFrame()

  season_data = nfl.import_pbp_data(years=seasons)
  pbp_data = pd.concat([pbp_data, season_data], axis=0)

  # Filter for New England Patriots offensive plays
  offensive_snaps = pbp_data[(pbp_data['posteam'] == team_abbrev) & (pbp_data['play_type'] != 'no_play')]
  # remove the non pass/run plays
  offensive_snaps = offensive_snaps[(offensive_snaps['play_type'] == 'run') | (offensive_snaps['play_type'] == 'pass')]
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
                'offense_personnel', 'defenders_in_box', 'defense_personnel', 'weather']]
  return offensive_snaps

team_plays_dict = {}
for team in teams:
  print(team)
  team_plays_dict[team] = process_team_data(team)

for team in teams:
  team_plays_dict[team].to_csv(team + "_data.csv")
