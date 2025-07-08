import pandas as pd

# Load the dataset
df = pd.read_csv("data/medical_insurance.csv")

# Dataset info
print("🧾 Dataset Info:")
print(df.info())

# Check for missing values
print("\n🧹 Null Value Check:")
print(df.isnull().sum())

# Save cleaned data
df.to_csv("data/cleaned_medical_insurance.csv", index=False)
print("\n✅ Cleaned data saved as 'cleaned_medical_insurance.csv'")
