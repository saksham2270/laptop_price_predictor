from re import T
import streamlit as st
import pickle
import numpy as np

pipe4 = pickle.load(open('pipe4.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title('Laptop Price Predictor')

company = st.selectbox('Brand',df['Company'].unique())

type = st.selectbox('Type',df['TypeName'].unique())

RAM = st.selectbox('RAM(in GB)',[1,2,4,6,8,12,16,32,64])

weight = st.number_input('Weight')

touchscreen = st.selectbox('Touchscreen',['No','Yes'])

ips = st.selectbox('IPS',['No','Yes'])

screen_size = st.number_input('Screen Size')

resolution = st.selectbox('Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2560x1600','2560x1440','2340x1440'])

cpu = st.selectbox('CPU',df['cpu'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

sdd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['gpu_brand'].unique())

os = st.selectbox('OS',df['OpSys'].unique())

if st.button('Predict Price'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5/screen_size


    query = np.array([company,type,RAM,os,weight,touchscreen,ips,ppi,cpu,hdd,sdd,gpu])
    query = query.reshape(1,12)
    st.title('Predicted Price of the laptop is ' +'Rs'+str(int(np.exp(pipe4.predict(query))[0])))




