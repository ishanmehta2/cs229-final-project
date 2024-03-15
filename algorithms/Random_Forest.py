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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss


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

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

# model_1 = LogisticRegression(solver='lbfgs', max_iter=10000)
# model_1.fit(X_train, Y_train)

# y_pred_1 = model_1.predict(X_test)

# accuracy_1 = accuracy_score(Y_test, y_pred_1)
# print(f'Model Accuracy: {accuracy_1:.4f}')

# cm = confusion_matrix(Y_test, y_pred_1)

# conf_matrix = confusion_matrix(Y_test, y_pred_1)
# # class_report = classification_report(Y_test, NN_prediction)

# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(outcome_run_pass), yticklabels=set(outcome_run_pass))
# plt.title('Confusion Matrix For Logistic Regression')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()

# PRINCIPAL COMPONENT ANALYSIS

numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns


ct = ColumnTransformer([
    ('scaler', StandardScaler(), numerical_cols)
], remainder='passthrough')  

scaled_data = ct.fit_transform(X)

# Use df
X_scaled = pd.DataFrame(scaled_data, columns=numerical_cols.tolist() + [col for col in X.columns if col not in numerical_cols])

pca = PCA(n_components=0.75)
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

X = pd.get_dummies(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.75)
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

print(len(X_pca))

# Plotting the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# 
# plt.figure(figsize=(12, 6))


# plt.subplot(1, 2, 1)
# plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
# plt.title('Explained Variance Ratio', fontsize=16, fontweight='bold')
# plt.xlabel('Principal Component', fontsize=14)
# plt.ylabel('Variance Ratio', fontsize=14)

# 
# plt.subplot(1, 2, 2)
# plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
# plt.title('Cumulative Explained Variance', fontsize=16, fontweight='bold')
# plt.xlabel('Number of Principal Components', fontsize=14)
# plt.ylabel('Cumulative Variance Ratio', fontsize=14)

# plt.tight_layout()
# plt.show()

loadings = pca.components_


loadings_df = pd.DataFrame(data=loadings, columns=X.columns, index=[f'PC{i}' for i in range(loadings.shape[0])])
feature_importances = np.sum(loadings**2, axis=0)
feature_ranking = np.argsort(feature_importances)[::-1]
list(X.columns)[34]
features = []
for feature_idx in feature_ranking:
    features.append(list(X.columns)[feature_idx])

data = pd.DataFrame()
for feature in feature_ranking:
    data[feature] = X_scaled[:,feature]

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, Y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

y_pred_probs = rf_model.predict_proba(X_test)

# Calculate log loss
log_loss_value = log_loss(Y_test, y_pred_probs)

print(f"Log Loss: {log_loss_value:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
report = classification_report(Y_test, y_pred)
print("Classification Report:")
print(report)

y_pred_1 = rf_model.predict(X_test)

accuracy_1 = accuracy_score(Y_test, y_pred_1)
print(f'Model Accuracy: {accuracy_1:.4f}')

cm = confusion_matrix(Y_test, y_pred_1)

conf_matrix = confusion_matrix(Y_test, y_pred_1)
# class_report = classification_report(Y_test, NN_prediction)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(outcome_run_pass), yticklabels=set(outcome_run_pass))
plt.title('Confusion Matrix For Random Forest')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
