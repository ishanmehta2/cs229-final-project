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
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
import scipy.stats as stats
from sklearn.metrics import log_loss
import statistics


big_data = pd.read_csv('/Users/ishan/Desktop/cs229-final-project/big_data.csv')

big_data = big_data.drop(['Unnamed: 0', 'game_id', 'game_date'], axis = 1)

bad_rows = []
for idx, row in big_data.iterrows():
    if idx % 2000 == 0: print(idx)
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

#teams = ['ARI', 'ATL']
teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV', 'LAC', 'LA', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']
accuracies = {}
test_accuracies = []
losses = []

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

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

    # NN = MLPClassifier(max_iter=100, activation='relu', hidden_layer_sizes=(100, 100))
    # NN.fit(X_train, Y_train)
    # NN_prediction = NN.predict(X_test)

    # print("Accuracy is", accuracy_score(Y_test, NN_prediction))
    # test_accuracies.append(accuracy_score(Y_test, NN_prediction))
    
    # NN_pred_probs = NN.predict_proba(X_test)

    # # Calculate log loss
    # NN_log_loss = log_loss(Y_test, NN_pred_probs)

    # losses.append(NN_log_loss)

    #print("Confusion Matrix:\n", confusion_matrix(Y_test, NN_prediction))

    #DROPOUT LAYER
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    encoded_Y = encoder.fit_transform(Y)  
    dummy_y = to_categorical(encoded_Y) 

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, dummy_y, test_size=0.3, random_state=1)

    # model = Sequential([
    #     Dense(128, input_dim=X_train.shape[1], activation='relu'),
    #     Dropout(0.1),
    #     Dense(64, activation='relu'),
    #     Dropout(0.1),
    #     Dense(Y_train.shape[1], activation='softmax') 
    # ])

    # 
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #             loss='categorical_crossentropy',
    #             metrics=['accuracy'])

    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # history = model.fit(X_train, Y_train, epochs=200, batch_size=20, 
    #                     validation_split=0.1, callbacks=[early_stopping])

    
    # 
    # loss, accuracy = model.evaluate(X_test, Y_test)
    # print(f'Test loss: {loss}, Test accuracy: {accuracy}')
    # test_accuracies.append(accuracy)
    # losses.append(loss)


    # CONVOLUTION LAYER
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(Y_train.shape[1], activation='softmax')  
    ])

    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    
    history = model.fit(X_train, Y_train, epochs=100, batch_size=24,
                        validation_split=0.1, callbacks=[early_stopping])

    
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')
    test_accuracies.append(accuracy)
    losses.append(loss)

# Accuracies statistics

# Maximum
max_value = max(test_accuracies)

# Minimum
min_value = min(test_accuracies)

# Median
median_value = statistics.median(test_accuracies)

# Average (Mean)
average_value = sum(test_accuracies) / len(test_accuracies)

print(f"Maximum: {max_value}")
print(f"Minimum: {min_value}")
print(f"Median: {median_value}")
print(f"Average: {average_value}")


# Loss Statistics
# Maximum
max_value = max(losses)

# Minimum
min_value = min(losses)

# Median
median_value = statistics.median(losses)

# Average (Mean)
average_value = sum(losses) / len(losses)

print(f"Maximum: {max_value}")
print(f"Minimum: {min_value}")
print(f"Median: {median_value}")
print(f"Average: {average_value}")



plt.figure(figsize=(20, 8))  
bars = plt.bar(teams, test_accuracies, color='skyblue')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.3f}', ha='center', va='bottom')

plt.xlabel('Teams')
plt.ylabel('Test Accuracy')
plt.title('Neural Network Accuracies By Team')
plt.ylim(0.4, 0.78)
plt.xticks(rotation=45, ha="right")  
plt.tight_layout()
plt.show()





# # Plotting
# plt.figure(figsize=(10, 6))  
# bars = plt.bar(teams, test_accuracies, color='skyblue')  

# 
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 3), va='bottom', ha='center')

# plt.xlabel('Team')  
# plt.ylabel('Test Accuracy')  
# plt.title('Neural Network Accuracies By Team (Dropout)')  
# plt.ylim(0.45, 0.76)  
# plt.xticks(rotation=45)  
# plt.tight_layout()  
# plt.show()  




# 
# conf_matrix = confusion_matrix(Y_test, NN_prediction)
# class_report = classification_report(Y_test, NN_prediction)

# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(outcome_run_pass), yticklabels=set(outcome_run_pass))
# plt.title('Confusion Matrix For Neural Network')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()
