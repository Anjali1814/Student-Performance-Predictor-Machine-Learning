# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:08:48 2024

@author: hp
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("diyaanjali.csv", encoding="ISO-8859-1")

# Drop unnecessary columns
X = df.drop(columns=['Grade', 'Field', 'Break Frequency'])
y = df['Grade'].values

# Preprocessing using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), X.columns)  # Apply OneHotEncoder to all columns
    ],
    remainder='passthrough'  # Keep any other columns as they are
)

# Fit and transform the data
X_encoded = preprocessor.fit_transform(X)

# Convert the sparse matrix to a dense format (array)
X_encoded_dense = X_encoded.toarray()

# Get feature names for the OneHotEncoder columns
ohe_columns = preprocessor.named_transformers_['onehot'].get_feature_names_out(X.columns)

# Create a DataFrame for the transformed data
X_encoded_df = pd.DataFrame(X_encoded_dense, columns=ohe_columns)

# Clean and encode target variable
y_series = pd.Series(y).str.strip().str.upper()
valid_grades = ['A', 'B', 'C', 'D', 'E', 'F']
y_filtered = y_series[y_series.isin(valid_grades)]

# Filter X based on the index of the valid grades in y
X_encoded_df = X_encoded_df.iloc[y_filtered.index]

# Create a LabelEncoder instance
le = LabelEncoder()
y_encoded = le.fit_transform(y_filtered)

# Print shapes for debugging
print("Shape of X_encoded_df:", X_encoded_df.shape)
print("Shape of y_encoded:", y_encoded.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded_df, y_encoded, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- SVM Model ---
svm = SVC(kernel='linear', C=1)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# SVM Performance
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# SVM Confusion Matrix
plt.figure(figsize=(10, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# --- KNN Model ---


# --- Naive Bayes Model ---
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Naive Bayes Performance
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nNaive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

# Naive Bayes Confusion Matrix
plt.figure(figsize=(10, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt='d', cmap='Blues')
plt.title('Naive Bayes Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# --- Visualizations for Outlier Detection ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='Break Duration', hue='Grade', data=df)  # Box plot to detect outliers
plt.title('Box Plot for Outlier Detection')
plt.xticks(rotation=45)
plt.show()

# Visualization of Study Duration by Grade
plt.figure(figsize=(10, 6))
sns.countplot(x='Study Duration', hue='Grade', data=df)
plt.title('Count of Study Duration by Grade')
plt.xticks(rotation=45)
plt.xlabel('Study Duration')
plt.ylabel('Count')
plt.legend(title='Grade')
plt.show()

