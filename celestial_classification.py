# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('star_classification_1.csv')

# Display the first few rows
print(df.head())

# Display dataset information
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Data processing
# Drop rows with missing values
df = df.dropna()

# Feature columns and target
features = ['u', 'g', 'r', 'i', 'z']
X = df[features]
y = df['class']

# Encode the target variable (if necessary)
y = pd.factorize(y)[0]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build and Train the Model
# Initialize RandomForestClassifier
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

#Evaluate the Model
# Predict on test data
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

#Hyperparameter Tuning
# Define parameter grid for GridSearch
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print best parameters
print(f'Best Parameters: {grid_search.best_params_}')

#Visualize Results
# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=df['class'].unique(), yticklabels=df['class'].unique())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Feature Importance Visualization
# Plot feature importance
importances = model.feature_importances_
feature_names = features

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Celestial Object Classification')
plt.show()
