# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import parallel_backend  # For parallel processing
import zipfile

# Set random seed for reproducibility
np.random.seed(42)

# --- Libraries and Data ---
# Load the data (assuming the file is a zipped CSV)
with zipfile.ZipFile('/kaggle/input/kobe-bryant-shot-selection/data.csv.zip', 'r') as z:
    z.extractall()
kobe = pd.read_csv('data.csv')

# --- Exploratory Data Analysis ---
# Scatter plot for shot locations using lat and lon
plt.figure(figsize=(10, 8))
sns.scatterplot(x='lon', y='lat', hue='shot_made_flag', data=kobe, alpha=0.5)
plt.title("Kobe's Shot Locations (lat vs lon)")
plt.show()

# Scatter plot for shot locations using loc_x and loc_y
plt.figure(figsize=(10, 8))
sns.scatterplot(x='loc_x', y='loc_y', hue='shot_made_flag', data=kobe, alpha=0.5)
plt.title("Kobe's Shot Locations (loc_x vs loc_y)")
plt.show()

# --- Feature Engineering ---
# Create new features
kobe['dist'] = np.sqrt(kobe['loc_x']**2 + kobe['loc_y']**2)
kobe['time_remaining'] = kobe['minutes_remaining'] * 60 + kobe['seconds_remaining']
kobe_f = kobe.drop(columns=['game_event_id', 'game_id', 'lat', 'lon', 'team_id', 'team_name', 'matchup', 'shot_id'])

# --- Data Preparation ---
# Split into training and testing sets based on shot_made_flag
train = kobe_f[kobe_f['shot_made_flag'].notna()]
test = kobe_f[kobe_f['shot_made_flag'].isna()]

# Extract test shot_ids for submission
test_id = kobe[kobe['shot_made_flag'].isna()]['shot_id']

# Define features and target
X_train = train.drop(columns=['shot_made_flag'])
y_train = train['shot_made_flag']
X_test = test.drop(columns=['shot_made_flag'])

# Identify categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = X_train.select_dtypes(exclude=['object']).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ])

# --- Random Forest Model ---
# Define the model
rf_model = RandomForestClassifier(n_estimators=800, random_state=42)

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])

# --- Hyperparameter Tuning ---
# Define parameter grid
param_grid = {
    'classifier__max_features': [1, len(X_train.columns)//2, len(X_train.columns)-1],  # mtry equivalent
    'classifier__min_samples_leaf': [1, 5, 10]  # min_n equivalent
}

# Set up 3-fold cross-validation
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Perform grid search with parallel processing
with parallel_backend('threading'):  # Use threading for parallelization
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1  # Use all available cores
    )
    grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best ROC AUC score:", grid_search.best_score_)

# --- Final Model and Predictions ---
# Fit the best model on the full training data
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predict probabilities on the test set
kobe_predictions_rf = best_model.predict_proba(X_test)[:, 1]  # Probability of class 1

# --- Submission File ---
# Create submission DataFrame
submission = pd.DataFrame({
    'shot_id': test_id,
    'shot_made_flag': kobe_predictions_rf
})

# Save to CSV
submission.to_csv('kobe_rf_submit.csv', index=False)
print("Submission file 'kobe_rf_submit.csv' created successfully.")

# --- Conclusion ---
print("The Random Forest model achieved a ROC AUC score of approximately", grid_search.best_score_)