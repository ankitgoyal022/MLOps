import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="ankitgoyal022/churn-model", filename="best_churn_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Churn Prediction
st.title("Tourism Prediction App")
st.write("Kindly enter the tourist details to check whether they are likely to churn.")

# Collect user input
CityTier = st.selectbox("CityTier", [1,2,3])
DurationOfPitch = st.selectbox("Duration Of Pitch", ["France", "Germany", "Spain"])
Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=30)
NumberOfPersonVisiting = st.number_input("Number Of PersonVisiting", value=12)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=0.0, value=10000.0)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1,2,3,4,5])
NumberOfTrips = st.number_input("Number Of Trips", min_value=1, value=1)
Passport = st.selectbox("Passport",  ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, value=1)
OwnCar = st.selectbox("OwnCar", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting",  min_value=0, max_value=1, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=1, value=1)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ['Salaried','Small Business','Large Business','Free Lancer'])
Gender = st.selectbox("Gender", ["Male ","Female","Fe Male"])
ProductPitched = st.selectbox("Product Pitched", [ "Basic","Deluxe","Standard","Super Deluxe","King"])
MaritalStatus = st.selectbox("Marital Status", ["Married","Divorced","Unmarried","Single"])
Designation = st.selectbox("Designation", ["Executive","Manager","Senior Manager","AVP","VP"])


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Age': Age,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips':NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting':NumberOfChildrenVisiting,
    'MonthlyIncome':MonthlyIncome,
    'TypeofContact':TypeofContact,
    'Occupation':Occupation,
    'Gender':Gender,
    'ProductPitched':ProductPitched,
    'MaritalStatus':MaritalStatus,
    'Designation':Designation
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "churn" if prediction == 1 else "not churn"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
