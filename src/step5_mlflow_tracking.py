# src/step5_mlflow_tracking.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Load dataset
df = pd.read_csv("data/cleaned_medical_insurance.csv")

# Encode categorical variables
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df['region'] = le_region.fit_transform(df['region'])

# Features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Medical_Insurance_Cost_Prediction")

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log parameters, metrics, model
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    mlflow.sklearn.log_model(model, "model")

    print("📊 MLflow run logged successfully!")
    print(f"📉 MSE: {mse:.2f}, 📈 R2 Score: {r2:.2f}")
