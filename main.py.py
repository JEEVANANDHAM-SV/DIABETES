import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\jeeva\Desktop\CAPSTONE\PROJECT 2\IMPLEMENT\diabetes_dataset.csv")


df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Polyuria'] = df['Polyuria'].map({'Yes': 1, 'No': 0})
df['Polydipsia'] = df['Polydipsia'].map({'Yes': 1, 'No': 0})
df['sudden weight loss'] = df['sudden weight loss'].map({'Yes': 1, 'No': 0})
df['weakness'] = df['weakness'].map({'Yes': 1, 'No': 0})
df['Polyphagia'] = df['Polyphagia'].map({'Yes': 1, 'No': 0})
df['Genital thrush'] = df['Genital thrush'].map({'Yes': 1, 'No': 0})
df['visual blurring'] = df['visual blurring'].map({'Yes': 1, 'No': 0})
df['Itching'] = df['Itching'].map({'Yes': 1, 'No': 0})
df['Irritability'] = df['Irritability'].map({'Yes': 1, 'No': 0})
df['delayed healing'] = df['delayed healing'].map({'Yes': 1, 'No': 0})
df['partial paresis'] = df['partial paresis'].map({'Yes': 1, 'No': 0})
df['muscle stiffness'] = df['muscle stiffness'].map({'Yes': 1, 'No': 0})
df['Alopecia'] = df['Alopecia'].map({'Yes': 1, 'No': 0})
df['Obesity'] = df['Obesity'].map({'Yes': 1, 'No': 0})
df['class'] = df['class'].map({'Positive': 1, 'Negative': 0})



st.title('Diabetes Checkup')
st.subheader('Training Data')
st.write(df.describe())

st.subheader('Visualization')
st.line_chart(df)


x = df.drop(['class'], axis = 'columns')
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)


st.subheader('Symptoms')
st.subheader('Give 1 for Male and Yes')
st.subheader('Give 0 for Female and No')
def user_report():
    Age = st.slider('Age', 18,100,18)
    Gender = st.selectbox('Gender',["1","0"])
    Polyuria = st.selectbox('Polyuria',["1","0"])
    Polydipsia = st.selectbox('Polydipsia',["1","0"])
    suddenweightloss = st.selectbox('sudden weightloss',["1","0"])
    weakness = st.selectbox('weakness',["1","0"])
    Polyphagia = st.selectbox('Polyphagia',["1","0"])
    Genitalthrush = st.selectbox('Genital thrush',["1","0"])
    visualblurring = st.selectbox('visual blurring',["1","0"])
    Itching = st.selectbox('Itching',["1","0"])
    Irritability = st.selectbox('Irritability',["1","0"])
    delayedhealing = st.selectbox('delayed healing',["1","0"])
    partialparesis = st.selectbox('partial paresis',["1","0"])
    musclestiffness = st.selectbox('muscle stiffness',["1","0"])
    Alopecia = st.selectbox('Alopecia',["1","0"])
    Obesity = st.selectbox('Obesity',["1","0"])

    user_report ={
        'Age':Age,
        'Gender':Gender,
        'Polyuria':Polyuria,
        'Polydipsia':Polydipsia,
        'sudden weight loss':suddenweightloss,
        'weakness':weakness,
        'Polyphagia':Polyphagia,
        'Genital thrush':Genitalthrush,
        'visual blurring':visualblurring,
        'Itching':Itching,
        'Irritability':Irritability,
        'delayed healing':delayedhealing,
        'partial paresis':partialparesis,
        'muscle stiffness':musclestiffness,
        'Alopecia':Alopecia,
        'Obesity':Obesity
    }
    report_data =pd.DataFrame(user_report, index=[0])
    return report_data

user_data = user_report()

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')

user_result = rf.predict(user_data)
st.subheader('Your Report: ')
output = ''
if user_result[0]==1:
    output = 'Positive'
else:
    output = 'Negative'

st.write(output)




