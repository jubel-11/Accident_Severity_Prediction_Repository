# 🛣️ Traffic Accident Severity Prediction Web App

A Machine Learning project that predicts the **severity level** of a traffic accident based on real-world roadway and environmental features.  
This system aims to support **road safety authorities** and **emergency responders** in making quicker decisions during accident scenarios.

---

## 📊 Dataset

**Source:** https://www.kaggle.com/datasets/s3programmer/road-accident-severity-in-india
The data set has been prepared from manual records of road traffic accidents for the years 2017–22.
The Road.csv dataset has 31 features with Target column as the Severity_level (Fatal, Serious, Slight) and 12316 instances of the accident.

---

## 🔧 Data Preprocessing

To handle categorical and numerical data effectively, the following steps were applied:
✔ Handled Missing values 
✔ **Label Encoding** for categorical features  
✔ **Feature Scaling** using MinMaxScaler
✔ The dataset was highly imbalanced, it was balanced using SMOTE (Synthethic Minority Oversampling Technique)
✔ **Feature Selection** using SelectFromModel and selected 20 features

---

## 🤖 Machine Learning Models Used

Multiple models were trained and compared and among them, XGBoost emerged as the most efficient model, offering strong handling of imbalanced data.
To classify the severity of the accident as **Fatal**, **Serious** or **Slight**, a **two-stage classification** system is developed:
- Stage 1: The model identifies whether an accident is Fatal or Not Fatal based on a threshold value. 
This stage achieved an **accuracy** of **83%**, successfully identifying fatal cases with a **recall** of **0.62**.
- Stage 2: For non-fatal cases, a secondary model classifies the severity into Slight Injury or Serious Injury. 
This stage achieved an accuracy of 70.6%, with a strong performance in recognizing slight injuries **(recall 0.76)** and reasonable identification of serious injuries **(recall 0.41)**.

Through this two-tiered approach, the model effectively balances precision and recall, ensuring reliable and interpretable predictions for traffic accident severity analysis.

---

## 🖥️ Model Deployment
Web app is created using Streamlit.
Try the Traffic Accident Severity Prediction
**👉 Live app :** https://accidentseverityprediction-q359fmaepzh92mhc45zvks.streamlit.app/

---

## Installation

# Clone the repository
git clone https://github.com/jubel-11/Accident_Severity_Prediction_Repository.git
cd Accident_Severity_Prediction_Repository

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

