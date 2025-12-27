import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("best_rf_model.pkl")

# Load the encoders
encoders = {
    'Sex': joblib.load("Sex_encoder.pkl"),
    'Housing': joblib.load("Housing_encoder.pkl"),
    'Saving accounts': joblib.load("Saving accounts_encoder.pkl"),
    'Checking account': joblib.load("Checking account_encoder.pkl")
}

st.title("Credit Risk Assessment")
st.write("Enter the following information to assess credit risk:")

# Streamlit input widgets
age = st.number_input("Age", min_value=18, max_value=70, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["Rent", "Own", "Free"])
saving_account = st.selectbox("Saving Account", ["little", "moderate", "rich", "quite rich", "no_saving_account"])
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "quite rich", "no_checking_account"])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration", min_value=1, value=12)

# Prepare input for the model
# Ensure all categorical inputs are converted to lowercase before encoding
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex.lower()])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing.lower()])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_account.replace(' ', '_').lower()])[0]], # Handle 'no saving account' etc.
    "Checking account": [encoders["Checking account"].transform([checking_account.replace(' ', '_').lower()])[0]], # Handle 'no checking account' etc.
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

if st.button("Predict Risk"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.write("Credit Risk: Good")
    else:
        st.write("Credit Risk: Bad")
