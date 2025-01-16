import streamlit as st
import pandas as pd
df=pd.read_excel('CompanyWise HR contact (1).xlsx')
st.selectbox('Company Name',df['Company'].values)
st.write(df[df['Name']])
st.write(df[df['Email']])
