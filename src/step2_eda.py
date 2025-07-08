# src/step2_eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_medical_insurance.csv")

# Set theme
sns.set(style="whitegrid")

# 1. Age vs Charges
plt.figure(figsize=(8, 5))
sns.scatterplot(x='age', y='charges', data=df, hue='smoker')
plt.title("Age vs Charges")
plt.savefig("outputs/age_vs_charges.png")
plt.close()

# 2. BMI vs Charges
plt.figure(figsize=(8, 5))
sns.scatterplot(x='bmi', y='charges', data=df, hue='smoker')
plt.title("BMI vs Charges")
plt.savefig("outputs/bmi_vs_charges.png")
plt.close()

# 3. Smoker vs Charges (Boxplot)
plt.figure(figsize=(6, 4))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title("Smoker vs Charges")
plt.savefig("outputs/smoker_vs_charges.png")
plt.close()

# 4. Region-wise Countplot
plt.figure(figsize=(6, 4))
sns.countplot(x='region', data=df)
plt.title("Count of Records per Region")
plt.savefig("outputs/region_count.png")
plt.close()

# 5. Correlation Heatmap (only numeric columns)
plt.figure(figsize=(6, 4))
numeric_df = df.select_dtypes(include=['int64', 'float64'])  # 👉 only numeric
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

print("✅ All EDA graphs saved in outputs/")
