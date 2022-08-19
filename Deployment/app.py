import numpy as np
import time
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('Deployment/model.pkl','rb'))
scaler = pickle.load(open('Deployment/scaler.pkl', 'rb'))

def predict(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, smoking):
    gender = 1 if gender == 'Male' else 0
    hypertension = 1 if hypertension == 'Yes' else 0
    heart_disease = 1 if heart_disease == 'Yes' else 0
    ever_married = 1 if ever_married == 'Yes' else 0

    if work_type == 'Government':
        work_type = 0
    elif work_type == 'Never worked':
        work_type = 1
    elif work_type == 'Private':
        work_type = 2
    elif work_type == 'Self employed':
        work_type = 3
    else:
        work_type = 4

    residence_type = 1 if residence_type == 'Urban' else 0
    smoking = 1 if smoking == 'Smoke' else 0

    features = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, smoking]])
    scaled = scaler.transform(features)
    pred = model.predict(scaled)
    return pred[0]

def main():
    html_temp = '''
    <h1 style="font-family: Trebuchet MS; padding: 12px; font-size: 48px; color: #6B705C; text-align: center;">
    <b>👨🏽Stroke Prediction👩🏽</b>
    <br><span style="color: #dcab6b; font-size: 20px">Deployed using Streamlit</span>
    </h1>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)

    st.write('Fill all the features below and click the predict button to see the result')

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.number_input('Age', step=1)
        avg_glucose_level = st.number_input('Avg Glucose Level', step=10)
    with col2:
        residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
        work_type = st.selectbox('Work Type', ['Private', 'Self employed', 'Children', 'Government', 'Never Worked'])
        smoking = st.selectbox('Smoking Status', ['Smoke', 'Never smoke'])
    with col3:
        heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
        ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
        hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
    
    if st.button("Predict"):
        st.markdown(
            """
            <style>
                .stProgress > div > div > div > div {
                    background-color: green;
                }
            </style>""",
            unsafe_allow_html=True,
        )

        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.001)
            my_bar.progress(percent_complete + 1)
        
        my_bar.empty()
        
        output = predict(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, smoking)
        if output == 1:
            st.error('You have high probability for having a stroke')
        else:
            st.success('You have low probability for having a stroke')

if __name__ == '__main__':
    main()