import streamlit as st
import numpy as np
import pickle

# Load your model
pickle_in = open("rf_model.pkl", "rb")
rf_model = pickle.load(pickle_in)

# Prediction function
def predict_loan_approval(loan_id, Gender, Married, Dependents, Education, Self_Employed, 
                           ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, 
                           Credit_History, Property_Area):
    # Create feature array including Loan_ID
    features = np.array([[loan_id, Gender, Married, Dependents, Education, Self_Employed, 
                          ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, 
                          Credit_History, Property_Area]])
    prediction = rf_model.predict(features)
    return prediction

# Main function to run the Streamlit app
def main():
    # Set page configuration
    st.set_page_config(page_title="Loan Approval Prediction System", page_icon="🏦", layout="wide")
    
    # Set page background image
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://yourdomain.com/path/to/background.jpg");
        background-size: cover;
        background-position: center;
    }

    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }

    [data-testid="stToolbar"] {
        right: 2rem;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    # Add an image with a smaller size
    st.image("https://yourdomain.com/path/to/home.jpg", width=250)
    
    # Add title and description
    st.title("🏦 Loan Approval Prediction System")
    st.markdown("""
    Welcome to the Loan Approval Prediction System. This application helps predict whether a loan application will be approved based on user inputs.  
    Please provide the required details below and click **Predict** to see the result.
    """)

    # Create two columns: one for the inputs and one for the info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Parameters")
        loan_id = st.number_input("Loan ID", min_value=0)
        gender = st.number_input("Gender (0 for Male, 1 for Female)", min_value=0, max_value=1, step=1)
        married = st.number_input("Married (0 for No, 1 for Yes)", min_value=0, max_value=1, step=1)
        dependents = st.number_input("Dependents (Enter as a number, e.g., 0, 1, 2, 3...)", min_value=0, max_value=3, step=1)
        education = st.number_input("Education (0 for Graduate, 1 for Not Graduate)", min_value=0, max_value=1, step=1)
        self_employed = st.number_input("Self Employed (0 for No, 1 for Yes)", min_value=0, max_value=1, step=1)
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0)
        loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
        credit_history = st.number_input("Credit History (0 for No, 1 for Yes)", min_value=0, max_value=1, step=1)
        property_area = st.number_input("Property Area (0 for Urban, 1 for Semiurban, 2 for Rural)", min_value=0, max_value=2, step=1)
    
    with col2:
        st.sidebar.header("About")
        st.sidebar.text("This app predicts loan approval based on input data using a pre-trained machine learning model.")
        st.sidebar.header("Contact")
        st.sidebar.text("For support, contact us at: support@example.com")

    result = ""

    # Display a button to trigger prediction
    if st.button("Predict"):
        result = predict_loan_approval(loan_id, gender, married, dependents, education, self_employed, 
                                       applicant_income, coapplicant_income, loan_amount, loan_amount_term, 
                                       credit_history, property_area)
        if result[0] == 1:
            st.success("🎉 Congratulations! Your loan is likely to be approved.")
        else:
            st.error("🚫 Unfortunately, your loan is likely to be rejected.")

if __name__ == '__main__':
    main()
