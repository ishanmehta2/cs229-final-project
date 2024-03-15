import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
import scipy.stats as stats
from tensorflow.keras.regularizers import l1_l2
import statistics


big_data = pd.read_csv('/Users/ishan/Desktop/cs229-final-project/big_data.csv')

big_data = big_data.drop(['Unnamed: 0', 'game_id', 'game_date'], axis = 1)

bad_rows = []
for idx, row in big_data.iterrows():
    if str(row['run_location']) == "nan" and str(row['pass_location']) == "nan":
        bad_rows.append(idx)
big_data.drop(bad_rows, inplace=True)

# cut down number of defense packages
dp_freq = dict(big_data['defense_personnel'].value_counts())
dp_freq[np.NaN] = 0
defense_package = []
for idx, row in big_data.iterrows():
    if dp_freq[row['defense_personnel']] < 450 or row['defense_personnel'] == np.NaN:
        defense_package.append(np.NaN)
    else:
        defense_package.append(row['defense_personnel'])
big_data = big_data.drop(['defense_personnel'],axis=1)
big_data['defense_personnel'] = defense_package

# cut down number of offense packages
op_freq = dict(big_data['offense_personnel'].value_counts())
op_freq[np.NaN] = 0
offense_package = []
for idx, row in big_data.iterrows():
    if op_freq[row['offense_personnel']] < 450 or row['offense_personnel'] == np.NaN:
        offense_package.append(np.NaN)
    else:
        offense_package.append(row['offense_personnel'])
big_data = big_data.drop(['offense_personnel'],axis=1)
big_data['offense_personnel'] = offense_package

team_data = big_data


# teams = ['ARI', 'ATL']
teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV', 'LAC', 'LA', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']
accuracies = {}
losses = {}

for team in teams: 
    relevant = team 
    team_games = team_data[(team_data['home_team'] == relevant) | (team_data['away_team'] == relevant)]
    print(len(team_games), relevant)

    X = team_games.drop(['play_sequence_num','pass_length', 'home_team', 'away_team', 'pass_location', 'run_location', 'play_type', 'surface'], axis=1)

    # Building y values 

    outcome_run_pass = []
    outcome_buckets = []
    for idx, row in team_games.iterrows():
        if row['play_type'] == 'pass':
            outcome_run_pass.append('pass')
            outcome_buckets.append('pass + ' + str(row['pass_location']))
        elif row['play_type'] == 'run':
            outcome_run_pass.append('run')
            outcome_buckets.append('run + ' + str(row['run_location']))


    X.fillna(0, inplace=True)
    X = pd.get_dummies(X)
    Y = outcome_run_pass
    Y = np.ravel(Y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = to_categorical(encoded_Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, dummy_y, test_size=0.3, random_state=1)

    num_samples, num_features = X_scaled.shape
    time_step_list = [6, 60, 200]
    for num_timesteps in time_step_list:
    

        total_elements = X_scaled.size
        num_features = len(X.columns)  
        for ts in range(1, total_elements):
            if total_elements % (ts * num_features) == 0:
                num_samples = total_elements // (ts * num_features)
                print(f"Possible reshape: ({num_samples}, {ts}, {num_features})")
                break

        X_scaled_reshaped = X_scaled.reshape((num_samples, ts, num_features))

        

        model = Sequential([
            LSTM(128, input_shape=(num_timesteps, num_features), return_sequences=False),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(dummy_y.shape[1], activation='softmax')  
        ])


        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

       
        history = model.fit(X_scaled_reshaped, dummy_y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

        
        loss, accuracy = model.evaluate(X_scaled_reshaped, dummy_y)
        print(f'Test loss: {loss}, Test accuracy: {accuracy}')
        accuracies[(team, num_timesteps)] = accuracy
        losses[(team, num_timesteps)] = loss


# ACCURACY 

accuracy_values = list(accuracies.values())

# Maximum accuracy
max_accuracy = max(accuracy_values)

# Minimum accuracy
min_accuracy = min(accuracy_values)

# Median accuracy
median_accuracy = statistics.median(accuracy_values)

# Average (mean) accuracy
average_accuracy = sum(accuracy_values) / len(accuracy_values)

print(f"Maximum accuracy: {max_accuracy}")
print(f"Minimum accuracy: {min_accuracy}")
print(f"Median accuracy: {median_accuracy}")
print(f"Average accuracy: {average_accuracy}")


# LOSS

loss_values = list(losses.values())

# Maximum
max_value = max(loss_values)

# Minimum
min_value = min(loss_values)

# Median
median_value = statistics.median(loss_values)

# Average (Mean)
average_value = sum(loss_values) / len(loss_values)

print(f"Maximum: {max_value}")
print(f"Minimum: {min_value}")
print(f"Median: {median_value}")
print(f"Average: {average_value}")



records = []
for team, timestep in accuracies:
    record = {'Team': team, 'Timestep': timestep, 'Accuracy': accuracies[(team, timestep)]}
    records.append(record)

df = pd.DataFrame(records)

pivot_df = df.pivot(index='Team', columns='Timestep', values='Accuracy')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_df, annot=True, fmt=".2%", cmap='viridis', cbar_kws={'label': 'Accuracy'})
plt.title('Team Accuracies Across Timesteps')
plt.xlabel('Timestep')
plt.ylabel('Team')

plt.xticks(rotation=45)

plt.show()

