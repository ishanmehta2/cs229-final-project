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


X = big_data.drop(['play_sequence_num','pass_length', 'home_team', 'away_team', 'pass_location', 'run_location', 'play_type', 'surface'], axis=1)

# Building y values 

outcome_run_pass = []
outcome_buckets = []
for idx, row in big_data.iterrows():
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

# NN Model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

model_1 = LogisticRegression(solver='lbfgs', max_iter=10000)
model_1.fit(X_train, Y_train)

y_pred_1 = model_1.predict(X_test)

accuracy_1 = accuracy_score(Y_test, y_pred_1)
print(f'Model Accuracy: {accuracy_1:.4f}')

# Building P_Test Logic
correct = []
incorrect = []
for i in range(len(Y_test)):
    if y_pred_1[i] == Y_test[i]:
        # correct
        correct.append(i)
    else:
        # incorrect
        incorrect.append(i)

misclassified_ex = X.iloc[incorrect]
correctly_classified = X.iloc[correct]


#relevant_dict = misclassified_ex['ydstogo'].value_counts()


p_values = []
keys = ['yardline_100', 'game_seconds_remaining', 'down', 'ydstogo', 'score_differential']


for i in range(len(keys)):
    relevant_dict = misclassified_ex[keys[i]].value_counts()
    accurate_dict = correctly_classified[keys[i]].value_counts()
    
    dicts = [relevant_dict, accurate_dict]
    
    for dict in dicts:
    
        weighted_sum = sum(key * frequency for key, frequency in dict.items())
        
        total_frequency = sum(dict.values)
        
        weighted_average = weighted_sum / total_frequency
        
        print(f"The weighted average of keys based on frequency is: {weighted_average}")
    
    
    relevant_sample = np.hstack([[key] * value for key, value in relevant_dict.items()])
    accurate_sample = np.hstack([[key] * value for key, value in accurate_dict.items()])
    
    ks_stat, ks_p_value = stats.ks_2samp(relevant_sample, accurate_sample)
    
    p_values.append(ks_stat)

    print(f"K-S statistic: {ks_stat}")
    
    
    if ks_p_value < 0.05:
        print("There is a statistically significant difference between the distributions.")
    else:
        print("There is no statistically significant difference between the distributions.")


# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(keys, p_values, color='skyblue')

plt.xlabel('Keys')
plt.ylabel('P-values')
plt.title('P-values for Kolmogorov-Smirnov Test Across Keys')
plt.axhline(0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
plt.legend()
plt.ylim(0, 0.06)  
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom', rotation=90)

plt.tight_layout()  
plt.show()

