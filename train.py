import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

# Sample dataset
data = {
    'age': [45,50,60,35,40,55,65,30],
    'cholesterol': [200,220,250,180,190,240,260,170],
    'bp': [120,130,140,110,115,135,145,105],
    'target': [0,1,1,0,0,1,1,0]
}

df = pd.DataFrame(data)

X = df.drop('target', axis=1)
y = df['target']

# Handle imbalance
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Scaling
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)

# Train model
model = RandomForestClassifier()
model.fit(X_res, y_res)

# Save
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Model ready ✅")