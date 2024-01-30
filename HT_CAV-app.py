# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/29 上午10:50
@Auth ： zy
@File ：HT_CAV-app.py
@IDE ：PyCharm
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap

st.write("""
# Adverse Events of Heart Transplant Recipients Prediction App

This app predicts the **Adverse Events** of heart transplant recipients!

*** 
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example XLSX input file](https://github.com/yz-hust/HT-Prediction-APP/blob/main/HT_Titles_Example.xlsx)
""")


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input xlsx file", type=["xlsx"])  # 上传文件控件
if uploaded_file is not None:
    input_df = pd.read_excel(uploaded_file)  # 待预测数据以Excel表格的形式上传
else:
    def user_input_features():
        # 默认待预测数据
        history_rejection = st.sidebar.selectbox('History of severe rejection (%)', ('0', '1'))  # 多选框控件 分类指标

        # 超声指标
        lvgls = st.sidebar.slider('LVGLS (%)', -30.0, 0.0, -15.6)  # 数字滑块控件
        rvfac = st.sidebar.slider('RVFAC (%)', 20.0, 50.0, 35.0)
        tapse = st.sidebar.slider('TAPSE (mm)', 0.0, 10.0, 2.8)
        lvef = st.sidebar.slider('LVEF (%)', 0.0, 100.0, 65.0)

        # 心肌酶
        bnp = st.sidebar.slider('BNP (pg/mL)', 0.0, 50.0, 15.0)
        ctni = st.sidebar.slider('cTnI (ug/L)', 0.0, 50.0, 15.0)

        # 血常规
        hemo = st.sidebar.slider('Hemoglobin (g/L)', 0.0, 50.0, 15.0)
        eryth = st.sidebar.slider('Erythrocyte counts (10*12/L)', 0.0, 50.0, 15.0)

        # 心肌酶
        bun = st.sidebar.slider('BUN (mmol/L)', 0.0, 50.0, 15.0)
        albumin = st.sidebar.slider('Albumin (g/L)', 0.0, 50.0, 15.0)
        creatinine = st.sidebar.slider('Creatinine (umol/L)', 0.0, 50.0, 15.0)

        # 其他
        heart_rate = st.sidebar.slider('Heart rate (bpm)', 0.0, 50.0, 15.0)

        donor_age = st.sidebar.slider('Donor age (years)', 0.0, 50.0, 15.0)
        cold_time = st.sidebar.slider('Cold ischemic time (hours)', 0.0, 50.0, 15.0)

        data = {'LVGLS': lvgls,
                'RVFAC': rvfac,
                'TAPSE': tapse,
                'LVEF': lvef,
                'BNP': bnp,
                'cTnI': ctni,
                'Hemoglobin': hemo,
                'Erythrocyte counts': eryth,
                'BUN': bun,
                'Albumin': albumin,
                'Creatinine': creatinine,
                'Heart rate': heart_rate,
                'History of severe rejection': history_rejection,
                'Donor age': donor_age,
                'Cold ischemic time': cold_time
                }
        features = pd.DataFrame(data, index=[0])
        return features


    input_df = user_input_features()


df = input_df


# Displays the user input features
st.subheader('Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
model = pickle.load(open(
    './RF_model_15.pkl',
    'rb'))

# # Apply model to make predictions
# prediction = load_clf.predict(df)

# st.subheader('Prediction')
# penguins_species = np.array(['no CAV', 'CAV'])
# st.write(penguins_species[prediction])

st.write('***')

if len(df) > 1:
    st.subheader('Multiple Patients Prediction Probability')

    select_patient = st.selectbox('Selecting a sample: ', range(len(df)))
    data = df.iloc[select_patient]
    if st.button("Predict"):
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)

        prediction_result = prediction_proba[select_patient][prediction[select_patient]]
        st.write("##### ***Predicted probaility of adverse events of heart transplant recipients is {}% !***".format(round(float(prediction_result) * 100, 2)))

        # 绘制SHAP结果图
        explainerTree = shap.TreeExplainer(model)
        shap_valuesTree = explainerTree.shap_values(df)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.force_plot(explainerTree.expected_value[1], shap_valuesTree[1][select_patient, :],
                        df.iloc[select_patient, :].astype(float),
                        matplotlib=True,
                        text_rotation=15)

        st.pyplot()

else:  # 单个预测样本
    st.subheader('Individual Patient Prediction Probability')
    # 添加预测按钮
    if st.button("Predict"):
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)

        prediction_result = prediction_proba[0][prediction]
        st.write("##### ***Predicted probaility of adverse events of heart transplant recipients is {}% !***".format(round(float(prediction_result) * 100, 2)))

        # 绘制SHAP结果图
        explainerTree = shap.TreeExplainer(model)
        shap_valuesTree = explainerTree.shap_values(df)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.force_plot(explainerTree.expected_value[1], shap_valuesTree[1][0, :],
                        df.iloc[0, :].astype(float),
                        matplotlib=True,
                        text_rotation=15)

        st.pyplot()
