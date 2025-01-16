import streamlit as st
import pandas as pd

# Load the data
df = pd.read_csv('CompanyWise.csv')

# Create a selectbox for company names
selected_company = st.selectbox('Company Name', df['Company'].values)

# Filter the DataFrame based on the selected company
company_info = df[df['Company'] == selected_company]

# Check if the company_info DataFrame is not empty
if not company_info.empty:
    # Display the Name and Email of the selected company
    st.write("Name:", company_info['Name'].values[0])
    st.write("Email:", company_info['Email'].values[0])
else:
    st.write("No information available for the selected company.")
