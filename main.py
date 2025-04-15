import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

@st.cache_resource
def load_model():
    with open("best_loan_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def preprocess_input(data):
    expected_features = [
        'person_age', 'person_gender', 'person_education', 'person_income',
        'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'previous_loan_defaults_on_file'
    ]
    input_df = pd.DataFrame([data])
    
    categorical_features = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
    
    le = LabelEncoder()
    for col in categorical_features:
        if col in input_df.columns:
            input_df[col] = le.fit_transform([input_df[col].iloc[0]])
    
    return input_df

def main():
    st.title("Loan Approval Prediction System")
    st.write("Enter the details below to predict loan approval status")
    
    if 'person_age' not in st.session_state:
        st.session_state.person_age = 30
    if 'person_gender' not in st.session_state:
        st.session_state.person_gender = "Male"
    if 'person_education' not in st.session_state:
        st.session_state.person_education = "Bachelor's"
    if 'person_income' not in st.session_state:
        st.session_state.person_income = 60000
    if 'person_emp_exp' not in st.session_state:
        st.session_state.person_emp_exp = 5
    if 'person_home_ownership' not in st.session_state:
        st.session_state.person_home_ownership = "RENT"
    if 'cb_person_cred_hist_length' not in st.session_state:
        st.session_state.cb_person_cred_hist_length = 10
    if 'credit_score' not in st.session_state:
        st.session_state.credit_score = 700
    if 'loan_amnt' not in st.session_state:
        st.session_state.loan_amnt = 10000
    if 'loan_intent' not in st.session_state:
        st.session_state.loan_intent = "PERSONAL"
    if 'loan_int_rate' not in st.session_state:
        st.session_state.loan_int_rate = 10.0
    if 'loan_percent_income' not in st.session_state:
        st.session_state.loan_percent_income = 20.0
    if 'previous_loan_defaults_on_file' not in st.session_state:
        st.session_state.previous_loan_defaults_on_file = "No"
    
    st.header("Personal Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.person_age, key="age_input")
        person_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.person_gender), key="gender_input")
        person_education = st.selectbox("Education", [
            "High School", "Bachelor's", "Master's", "PhD", 
            "Associate's", "Some College", "Other"
        ], index=[
            "High School", "Bachelor's", "Master's", "PhD", 
            "Associate's", "Some College", "Other"
        ].index(st.session_state.person_education), key="education_input")
        person_income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=st.session_state.person_income, key="income_input")
    
    with col2:
        person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=50, value=st.session_state.person_emp_exp, key="emp_exp_input")
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], 
                                            index=["RENT", "MORTGAGE", "OWN", "OTHER"].index(st.session_state.person_home_ownership), key="ownership_input")
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, 
                                                    value=st.session_state.cb_person_cred_hist_length, key="cred_hist_input")
        credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=st.session_state.credit_score, key="credit_score_input")
    
    st.header("Loan Details")
    
    col3, col4 = st.columns(2)
    
    with col3:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, max_value=100000, value=st.session_state.loan_amnt, key="loan_amnt_input")
        loan_intent = st.selectbox("Loan Intent", [
            "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", 
            "HOME_IMPROVEMENT", "DEBT_CONSOLIDATION", "OTHER"
        ], index=[
            "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", 
            "HOME_IMPROVEMENT", "DEBT_CONSOLIDATION", "OTHER"
        ].index(st.session_state.loan_intent), key="loan_intent_input")
        
    with col4:
        loan_int_rate = st.slider("Interest Rate (%)", min_value=1.0, max_value=30.0, value=st.session_state.loan_int_rate, step=0.1, key="int_rate_input")
        loan_percent_income = st.slider("Loan Percent of Income (%)", min_value=1.0, max_value=100.0, 
                                        value=st.session_state.loan_percent_income, step=0.1, key="percent_income_input")
        
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File", ["Yes", "No"], 
                                                 index=["Yes", "No"].index(st.session_state.previous_loan_defaults_on_file), key="defaults_input")
    
    st.session_state.person_age = person_age
    st.session_state.person_gender = person_gender
    st.session_state.person_education = person_education
    st.session_state.person_income = person_income
    st.session_state.person_emp_exp = person_emp_exp
    st.session_state.person_home_ownership = person_home_ownership
    st.session_state.cb_person_cred_hist_length = cb_person_cred_hist_length
    st.session_state.credit_score = credit_score
    st.session_state.loan_amnt = loan_amnt
    st.session_state.loan_intent = loan_intent
    st.session_state.loan_int_rate = loan_int_rate
    st.session_state.loan_percent_income = loan_percent_income
    st.session_state.previous_loan_defaults_on_file = previous_loan_defaults_on_file
    
    input_data = {
        'person_age': person_age,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': person_home_ownership,
        'loan_amnt': loan_amnt,
        'loan_intent': loan_intent,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file
    }
    
    st.header("Prediction")
    if st.button("Predict Loan Status"):
        try:
            model = load_model()
            processed_input = preprocess_input(input_data)
            prediction = model.predict(processed_input)[0]
            
            if prediction == 1:
                st.success("Loan Approved")
                st.write("The model predicts that this loan is likely to be approved.")
            else:
                st.error("Loan Denied")
                st.write("The model predicts that this loan is likely to be denied.")
            
            try:
                prob = model.predict_proba(processed_input)[0]
                st.write(f"Approval probability: {prob[1]:.2%}")
                
                st.progress(float(prob[1]))
            except:
                pass
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("Check that the feature names match what the model expects.")
    
    st.header("Test Cases")
    col5, col6 = st.columns(2)
    
    with col5:
        if st.button("Load Test Case 1 (Likely Approval)"):
            st.session_state.person_age = 35
            st.session_state.person_gender = "Male"
            st.session_state.person_education = "Bachelor's"
            st.session_state.person_income = 85000
            st.session_state.person_emp_exp = 10
            st.session_state.person_home_ownership = "MORTGAGE"
            st.session_state.loan_amnt = 15000
            st.session_state.loan_intent = "HOME_IMPROVEMENT"
            st.session_state.loan_int_rate = 8.5
            st.session_state.loan_percent_income = 17.6
            st.session_state.cb_person_cred_hist_length = 12
            st.session_state.credit_score = 780
            st.session_state.previous_loan_defaults_on_file = "No"
            st.rerun()
    
    with col6:
        if st.button("Load Test Case 2 (Likely Denial)"):
            st.session_state.person_age = 22
            st.session_state.person_gender = "Female"
            st.session_state.person_education = "High School"
            st.session_state.person_income = 28000
            st.session_state.person_emp_exp = 1
            st.session_state.person_home_ownership = "RENT"
            st.session_state.loan_amnt = 25000
            st.session_state.loan_intent = "PERSONAL"
            st.session_state.loan_int_rate = 1.00
            st.session_state.loan_percent_income = 1.00
            st.session_state.cb_person_cred_hist_length = 6
            st.session_state.credit_score = 580
            st.session_state.previous_loan_defaults_on_file = "Yes"
            st.rerun()

if __name__ == "__main__":
    main()