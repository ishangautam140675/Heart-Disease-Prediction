HEART DISEASE PREDICTION USING MACHINE LEARNING (BINARY CLASSIFICATION WITH IMBALANCE HANDLING)
1. Introduction

Heart disease prediction is an important application of Machine Learning in the healthcare sector. The main objective of this project is to predict whether a person is likely to have heart disease based on various medical attributes such as age, cholesterol level, blood pressure, etc. This helps doctors in early diagnosis and treatment.

In this project, we use a classification algorithm to predict the presence (1) or absence (0) of heart disease. Since medical datasets are often imbalanced, imbalance handling techniques are also applied to improve model performance.

2. Objective

The main objectives of this project are:

To understand binary classification
To predict heart disease using patient data
To handle imbalanced datasets
To build a reliable prediction model
3. Technology Used
Programming Language: Python
Libraries Used:
Pandas (for data handling)
NumPy (for numerical operations)
Matplotlib (for visualization)
Scikit-learn (for machine learning model)
Imbalanced-learn (for handling imbalance)
Flask (for web application)
4. Dataset Description

The dataset contains the following features:

Age
Sex
Chest Pain Type
Resting Blood Pressure
Cholesterol
Fasting Blood Sugar
Rest ECG
Maximum Heart Rate
Exercise Induced Angina
Oldpeak
Slope
Number of Major Vessels
Thal
Target (0 = No Disease, 1 = Disease)

Sample Data:

Age	Sex	Cholesterol	MaxHR	Target
63	1	233	150	1
45	0	204	172	0
5. Methodology
Step 1: Data Collection

The dataset is collected in CSV format.

Step 2: Data Preprocessing
Handling missing values
Encoding categorical variables
Feature selection
Step 3: Handling Imbalanced Data

Since the dataset may have unequal classes:

SMOTE (Synthetic Minority Oversampling Technique) is used
It balances the dataset by generating synthetic samples
Step 4: Splitting Dataset

The dataset is divided into:

Training Data (80%)
Testing Data (20%)
Step 5: Model Training

A classification model such as Logistic Regression or Random Forest is trained.

Step 6: Prediction

The model predicts whether a person has heart disease or not.

Step 7: Evaluation

Model performance is evaluated using:

Accuracy
Confusion Matrix
Precision and Recall
6. Algorithm Used: Logistic Regression / Random Forest
Logistic Regression:

Used for binary classification problems.

Equation:
P = 1 / (1 + e^(-z))

Random Forest:

Combines multiple decision trees for better accuracy.

7. Implementation (Code)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv("heart_data.csv")

# Features and target
X = data.drop("target", axis=1)
y = data["target"]

# Handle imbalance
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
8. Results

The model successfully predicts whether a person has heart disease or not.
After applying SMOTE, the model performance improves, especially in detecting minority class cases.

9. Advantages
Helps in early detection of heart disease
Handles imbalanced data effectively
Provides reliable predictions
Useful in healthcare decision-making
10. Limitations
Depends on quality of medical data
Requires proper preprocessing
Cannot replace medical professionals
11. Conclusion

This project demonstrates how machine learning can be used in healthcare to predict heart disease. By using classification algorithms and handling imbalanced data, the model achieves better accuracy and reliability.

12. Future Scope
Use deep learning models
Integrate with hospital systems
Deploy as a web application
Improve dataset size and quality
