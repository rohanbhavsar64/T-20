import streamlit as st
import pandas as pd
df=pd.read_csv('CompanyWise.csv')
st.selectbox('Company Name',df['Company'].values)
st.write(df[df['Name']].values[0])
st.write(df[df['Email']].values[0])
