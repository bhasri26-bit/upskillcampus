# Diabetes Prediction using Machine Learning
# Project: Data Science and Machine Learning

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset can be kept in the same folder as "diabetes.csv".
# If it is not present, the code downloads the commonly used Pima Indians Diabetes dataset.
LOCAL_FILE = "diabetes.csv"
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

try:
    data = pd.read_csv(LOCAL_FILE)
    # Add column names if the local file has no header.
    if "Outcome" not in data.columns:
        data.columns = columns
except FileNotFoundError:
    data = pd.read_csv(DATA_URL, names=columns)

print("First five records:")
print(data.head())

print("\nDataset shape:", data.shape)
print("\nMissing values:")
print(data.isnull().sum())

# In this dataset, zero is not a meaningful value for several medical
# measurements. Replace those zeros with NaN and fill with the median.
zero_columns = [
    "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI"
]

data[zero_columns] = data[zero_columns].replace(0, np.nan)
data[zero_columns] = data[zero_columns].fillna(data[zero_columns].median())

# Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Example prediction
# Order:
# Pregnancies, Glucose, BloodPressure, SkinThickness,
# Insulin, BMI, DiabetesPedigreeFunction, Age

sample = np.array([[2, 140, 80, 30, 100, 32.0, 0.45, 35]])
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0][1] * 100

print("\nSample Prediction:")
if prediction == 1:
    print("Result: Diabetes is predicted.")
else:
    print("Result: Diabetes is not predicted.")

print("Estimated probability of diabetes:",
      round(probability, 2), "%")
