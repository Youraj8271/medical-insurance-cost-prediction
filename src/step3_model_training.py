# src/step3_model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Load cleaned data
df = pd.read_csv("data/cleaned_medical_insurance.csv")

# Encode categorical variables
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df['region'] = le_region.fit_transform(df['region'])

# Define features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📉 Mean Squared Error: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Save model
os.makedirs("model", exist_ok=True)
with open("model/insurance_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved to model/insurance_model.pkl")
