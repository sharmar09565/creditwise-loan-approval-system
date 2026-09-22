# CreditWise Loan Approval System

CreditWise is a machine learning-based loan approval prediction system that predicts whether a loan application is likely to be approved based on applicant, financial, and loan-related information.

## Features

* Loan approval prediction using Machine Learning
* Data preprocessing and handling of missing values
* Categorical feature encoding
* Feature scaling
* Interactive Streamlit web application
* Logistic Regression model for prediction

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib
* Jupyter Notebook

## Input Information

The application uses details such as:

* Applicant income
* Coapplicant income
* Age
* Credit score
* Employment status
* Education level
* Existing loans
* DTI ratio
* Savings
* Collateral value
* Loan amount
* Loan term
* Loan purpose
* Property area

## Project Structure

```text
CreditWise Loan Approval System/
│
├── app.py
├── creditwise_loan_system.ipynb
├── loan_approval_data.csv
├── loan_model.pkl
├── scaler.pkl
├── onehot_encoder.pkl
├── education_encoder.pkl
├── num_imputer.pkl
├── cat_imputer.pkl
└── .gitignore
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/sharmar09565/creditwise-loan-approval-system.git
```

Navigate to the project folder:

```bash
cd creditwise-loan-approval-system
```

Install the required libraries:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

Run the application:

```bash
streamlit run app.py
```

## Model

The project uses **Logistic Regression** to predict loan approval after preprocessing, encoding, and scaling the input data.

## Author

**Rohit Sharma**

B.Tech CSE | Data Science & Machine Learning
