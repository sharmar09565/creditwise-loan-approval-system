import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessing objects
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("onehot_encoder.pkl")
education_encoder = joblib.load("education_encoder.pkl")
num_imp = joblib.load("num_imputer.pkl")


# Page settings
st.set_page_config(
    page_title="CreditWise Loan Approval",
    page_icon="💰",
    layout="wide"
)

st.title("CreditWise Loan Approval System")
st.write("Enter the applicant details below to predict loan approval.")


# Applicant Information
st.header("Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=5000.0
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single"]
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=0
    )

with col3:
    education_level = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

    employment_status = st.selectbox(
        "Employment Status",
        ["Salaried", "Self-employed", "Unemployed"]
    )

    employer_category = st.selectbox(
        "Employer Category",
        ["Private", "Government", "Self-employed"]
    )


# Financial Information
st.header("Financial Information")

col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=900.0,
        value=650.0
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=20,
        value=0
    )

with col2:
    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.30
    )

    savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=10000.0
    )

with col3:
    collateral_value = st.number_input(
        "Collateral Value",
        min_value=0.0,
        value=30000.0
    )


# Loan Information
st.header("Loan Information")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=20000.0
    )

with col2:
    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1,
        max_value=360,
        value=60
    )

with col3:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Personal", "Home", "Car", "Education", "Business"]
    )

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)


# Prediction
st.divider()

if st.button("Predict Loan Approval", type="primary"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "applicant_income": [applicant_income],
        "coapplicant_income": [coapplicant_income],
        "employment_status": [employment_status],
        "age": [age],
        "marital_status": [marital_status],
        "dependents": [dependents],
        "credit_score": [credit_score],
        "existing_loans": [existing_loans],
        "dti_ratio": [dti_ratio],
        "savings": [savings],
        "collateral_value": [collateral_value],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "loan_purpose": [loan_purpose],
        "property_area": [property_area],
        "education_level": [education_level],
        "gender": [gender],
        "employer_category": [employer_category]
    })

    # Numerical columns
    numeric_columns = [
        "applicant_income",
        "coapplicant_income",
        "age",
        "dependents",
        "credit_score",
        "existing_loans",
        "dti_ratio",
        "savings",
        "collateral_value",
        "loan_amount",
        "loan_term"
    ]

    # Categorical columns
    categorical_columns = [
        "employment_status",
        "marital_status",
        "loan_purpose",
        "property_area",
        "gender",
        "employer_category"
    ]

    # Apply numerical imputer
    input_data[numeric_columns] = num_imp.transform(
        input_data[numeric_columns]
    )

    # Encode education level
    input_data["education_level"] = education_encoder.transform(
        input_data["education_level"]
    )

    # One-hot encode categorical columns
    encoded = ohe.transform(
        input_data[categorical_columns]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(categorical_columns),
        index=input_data.index
    )

    # Combine encoded and remaining columns
    input_data = pd.concat(
        [
            input_data.drop(columns=categorical_columns),
            encoded_df
        ],
        axis=1
    )

    # Make sure columns are in the same order as training data
    input_data = input_data[scaler.feature_names_in_]

    # Scale the data
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Display result
    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")