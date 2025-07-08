# 🏥 Medical Insurance Cost Prediction using Machine Learning

📌 This project aims to predict the medical insurance cost for individuals using various features like age, sex, BMI, children, smoking status, and region.
Built using Python, Streamlit, Scikit-learn, and MLflow as part of the GUVI Capstone Project.

## 📁 Project Structure
📦 medical-insurance-cost-prediction/
├── data/                     # Original & Cleaned CSV file
├── model/                    # Trained ML model (.pkl)
├── outputs/                  # EDA plots (images)
├── src/                      # All Python scripts
│   ├── step1_data_cleaning.py
│   ├── step2_eda.py
│   ├── step3_model_training.py
│   ├── step4_streamlit_app.py
│   └── step5_mlflow_tracking.py
├── requirements.txt          # Python dependencies
├── README.md                 # This file


## 🚀 How to Run This Project

# Clone the repository
git clone https://github.com/Youraj8271/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction

# Install all dependencies
pip install -r requirements.txt

# Launch the Streamlit Web App
streamlit run src/step4_streamlit_app.py

| Details             | Value              |
|---------------------|--------------------|
| Algorithm           | Linear Regression  |
| R² Score            | 0.74               |
| Mean Squared Error  | ₹ 39.9 Lakhs       |
| Model Tracking      | MLflow             |


🔬 MLflow Tracking

⭐ Track experiments, metrics & versions:


# Step 1: Log the model

⭐ python src/step5_mlflow_tracking.py

# Step 2: Launch MLflow UI

⭐ mlflow ui

⭐ Open browser: http://127.0.0.1:5000

🌐 Streamlit App Preview
🧍‍♂️ Enter personal attributes
💡 Click “Predict Insurance Cost”
💰 Get your insurance estimate in seconds

🧰 Technologies Used

⭐ Python

⭐ Pandas, Numpy

⭐ Scikit-learn

⭐ Seaborn, Matplotlib

⭐ Streamlit

⭐ MLflow

👨‍💻 Developed By
 Youraj Kumar
💼 Data Scientist | Machine Learning Engineer
📬 youraj_2412res154@iitp.ac.in
🔗 GitHub: Youraj8271

📌 Project Status
✅ Completed: Full pipeline implemented
🧠 Includes data cleaning, EDA, model training, deployment, experiment tracking
🛠️ Open to improvements and scaling






