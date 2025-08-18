# App launch and contents file
'''
This is the file that you need to get Streamlit to run, and then all of the listed
pages will get rendered.

Run the following command:

python -m streamlit run app.py

''' 

import streamlit as st

st.set_page_config(page_title="Code Club Streamlit Demo", page_icon="🤖", layout="wide")

home = st.Page("pages/home.py", title="Home", icon="🏠", default=True)
explore = st.Page("pages/exploration.py", title="Data Exploration", icon="📊")
forecast = st.Page("pages/forecast.py", title="Simple Demo", icon="📈")

pg = st.navigation({
    "Home": [home],
    "Exploration": [explore],
    "Forecast":[forecast]
})

pg.run()