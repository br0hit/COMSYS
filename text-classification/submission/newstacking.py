import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.ensemble import StackingClassifier

SEED=123

file_name = '../preprocessing/vectorization/bm25_vect_with_custom1.npz'

# Load the .npz file
loaded_data = np.load(file_name)

# Access and load individual arrays from the loaded data
X_train = loaded_data['X_train']
X_test_ = loaded_data['X_test']

with open('../data/encoded_labels.txt', 'r') as file:
    y_train = np.array([int(line.strip()) for line in file])


base_models = [
    ('RandomForest', RandomForestClassifier(random_state=SEED, n_estimators=100)),
    ('Logistic', LogisticRegression(random_state=SEED)),
    ('GradientBoost', GradientBoostingClassifier(random_state=SEED)),
    ('XGB', XGBClassifier(random_state=SEED)),
    ('SVM', SVC(random_state=SEED)) ,   
]

# Create a stacking ensemble with a meta-model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier(random_state=SEED, n_estimators=100))

# Train the stacking ensemble and prediction

stacking_model.fit(X_train, y_train)
y_pred_test = stacking_model.predict(X_test_)

# Assuming you have predicted labels in a NumPy array called 'predicted_labels'
# You also have a dictionary that maps original labels to encoded labels
label_encoding = {0:'Anger/ Intermittent Explosive Disorder',
    1:'Anxiety Disorder',
    2:'Depression',
    3:'Narcissistic Disorder',
    4:'Panic Disorder'}

# Create serial numbers (ids) starting from 1
ids = np.arange(0, len(y_pred_test) )

# Encode the predicted labels
encoded_labels = [label_encoding[label] for label in y_pred_test]

# Create a list of rows where each row contains id and encoded label
data = list(zip(ids, encoded_labels))

# Save the data to a text file with 'id' and 'label' columns
with open('output_file_ff.csv', 'w') as file:
    file.write('id,label\n')  # Write header
    for row in data:
        file.write(f'{row[0]},{row[1]}\n')  # Write id and label separated by a comma