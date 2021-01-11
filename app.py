from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('rds_diploma')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))

    st.sidebar.info('Heart Failure Prediction')

    st.title("Heart Failure Prediction")

    if add_selectbox == 'Online':
        
        col1, col2 = st.beta_columns(2)
        
        with col1:
            age = st.number_input('Age', min_value=1, max_value=100, value=60)
            anm = st.selectbox('Anaemia', ['Yes', 'No'])
            hbp = st.selectbox('High Blood_Pressure', ['Yes', 'No'])
            
        with col2: 
            ejf = st.number_input('Ejection Fraction', min_value=0, max_value=100, value=35)
            scr = st.number_input('Serum Creatinine', min_value=0, max_value=10, value=2)

        output = ""

        input_dict = {'age' : age, 'anm' : anm, 'ejf' : ejf, 'hbp' : hbp, 'scr' : scr}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            if output == 1:
                outstr='Присутствует' 
            else outstr='Отсуствует'

        st.success('{} вероятность смерти пациента '.format(outstr))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model, data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()