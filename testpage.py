import streamlit as st
import numpy as np

# Create dummy data
data = np.random.rand(100, 50)

# Set page layout to wide
st.set_page_config(layout="wide")

# Create an HTML table
html = '<table border="1" style="width:100%; white-space: nowrap;">'
html += '<tr>' + ''.join([f'<th>Column {i+1}</th>' for i in range(data.shape[1])]) + '</tr>'

for row in data:
    html += '<tr>' + ''.join([f'<td>{cell:.2f}</td>' for cell in row]) + '</tr>'

html += '</table>'

# Display the HTML table
st.markdown(html, unsafe_allow_html=True)
