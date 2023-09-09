


import numpy as np

file_name = '../preprocessing/vectorization/bm25_vect_with_custom1.npz'
# Load the .npz file
loaded_data = np.load(file_name)
# Access and load individual arrays from the loaded data
X_train = loaded_data['X_train']
X_test_ = loaded_data['X_test']


from sklearn.model_selection import train_test_split


with open('../data/encoded_labels.txt', 'r') as file:
    y_train = np.array([int(line.strip()) for line in file])

# print(y_train.shape)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
SEED=123


#stacking---------


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

# 1. Vectorization (BM25) - You already have this part

# 2. Split the data
# You've already done this step


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=SEED)


# 3. Initialize the models
rf = RandomForestClassifier(random_state=SEED)
lr = LogisticRegression(random_state=SEED)
gb = GradientBoostingClassifier(random_state=SEED)

# 4. Train and evaluate each model
models = [rf, lr, gb]
stacked_train = []

for model in models:
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the training data
    y_pred_train = model.predict(X_train)
    
    # Evaluate the performance of the model
    accuracy = accuracy_score(y_train, y_pred_train)
    print(f"{model.__class__.__name__} - Training Accuracy score:", accuracy)
    
    # Append the predictions to the stacked_train list
    stacked_train.append(y_pred_train)

# 5. Stack the predictions
stacked_train = np.column_stack(stacked_train)

# 6. Initialize a meta-model (e.g., Random Forest, Logistic Regression, etc.) for stacking
meta_model = RandomForestClassifier(random_state=SEED)

# 7. Train the meta-model on the stacked predictions
meta_model.fit(stacked_train, y_train)

# 8. Make predictions on the test set for each base model
stacked_test = []
stacked_test_ = []

for model in models:
    # Predict on the test data
    y_pred_test = model.predict(X_test)
    y_pred_test_ = model.predict(X_test_)
    
    # Append the predictions to the stacked_test list
    stacked_test.append(y_pred_test)
    stacked_test_.append(y_pred_test_)


# 9. Stack the test predictions
stacked_test = np.column_stack(stacked_test)
stacked_test_ = np.column_stack(stacked_test_)

# 10. Make predictions using the meta-model
y_pred_meta = meta_model.predict(stacked_test)
# y_pred_meta_ = meta_model.predict(stacked_test_)

# 11. Evaluate the performance of the stacked model
stacked_accuracy = accuracy_score(y_test, y_pred_meta)
print("Stacked Model - Testing Accuracy score:", stacked_accuracy)


# Split the remaining 20% of the data into training and testing subsets
X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(stacked_train, y_train, test_size=0.2, random_state=42)

# Initialize the final model (e.g., Random Forest) and specify hyperparameters
final_model = RandomForestClassifier(random_state=SEED, n_estimators=100)  # You can adjust hyperparameters as needed

# Train the final model on the remaining 20% of the data
final_model.fit(X_final_train, y_final_train)

# Predict on the test subset of the remaining data
y_final_pred_test = final_model.predict(X_final_test)

# Evaluate the performance of the final model on this test subset
final_model_accuracy = accuracy_score(y_final_test, y_final_pred_test)
print("Final Model - Testing Accuracy score:", final_model_accuracy)

# X_test = bm25_[num_lines_file:][:]
y_pred_test = final_model.predict(stacked_test_)









# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score

# # Initialize the Random Forest classifier
# rf = RandomForestClassifier(random_state=SEED)

# # Define a grid of hyperparameters to search
# param_grid = {
#     'n_estimators': [100],       # Number of trees in the forest
#     # 'max_depth': [None, 10, 20, 30],       # Maximum depth of each tree
#     # 'min_samples_split': [2, 5, 10],       # Minimum samples required to split a node
#     # 'min_samples_leaf': [1, 2, 4],         # Minimum samples required for a leaf node
#     # 'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider for the best split
# }

# # Create a GridSearchCV object with the Random Forest classifier and parameter grid
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# # Fit the grid search to the training data
# grid_search.fit(X_train, y_train)

# # Get the best hyperparameters from the grid search
# best_params = grid_search.best_params_

# # Use the best hyperparameters to create a new Random Forest classifier
# best_rf = RandomForestClassifier(random_state=SEED, **best_params)

# # Fit the best model to the training data
# best_rf.fit(X_train, y_train)

# # # Predict on the training and testing data using the best model
# y_pred_train_rf = best_rf.predict(X_train)
# y_pred_test_rf = best_rf.predict(X_test)

# # Evaluate the performance of the best Random Forest model
# print("Best Model - Training Accuracy score:", accuracy_score(y_train, y_pred_train_rf))
# print("Best Model - Testing Accuracy score:", accuracy_score(y_test, y_pred_test_rf))


#--------  saving output file
import numpy as np

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
with open('output_file_stacked1.csv', 'w') as file:
    file.write('id,label\n')  # Write header
    for row in data:
        file.write(f'{row[0]},{row[1]}\n')  # Write id and label separated by a comma