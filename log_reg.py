import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import seaborn as sns

dfs = []
folder_path = '/Users/ohmpatel/Documents/GitHub/cs229-final-project/updated_updated_team_data'
for filename in os.listdir('./updated_updated_team_data'):
  print(filename)
  file_path = os.path.join(folder_path, filename)
  df = pd.read_csv(file_path)
  dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
for idx, row in data.iterrows():
  if str(row['run_location']) == "nan" and str(row['pass_location']) == "nan":
    data.drop(idx, inplace=True)

data.to_csv('combined_data.csv')
 
outcome_run_pass = []
outcome_buckets = []
for idx, row in data.iterrows():
  if row['play_type'] == 'pass':
    outcome_run_pass.append('pass')
    outcome_buckets.append('pass + ' + str(row['pass_location']))
  elif row['play_type'] == 'run':
    outcome_run_pass.append('run')
    outcome_buckets.append('run + ' + str(row['run_location']))


X = data.drop(['Unnamed: 0', 'game_id', 'game_date', 'pass_length', 'pass_location', 'run_location', 'play_type'], axis=1)
X = pd.get_dummies(X)
X = np.array(X)


result_d = {}
i = 0
for outcome in set(outcome_buckets):
  if outcome not in result_d:
    result_d[outcome] = i
    i += 1

y = []
for outcome in outcome_buckets:
  y.append(result_d[outcome])
y = np.array(y)

X = np.array(X)
X = np.nan_to_num(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the logistic regression model with softmax activation (multi_class='multinomial')
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)

# Train the model
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=result_d.keys(), yticklabels=result_d.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()