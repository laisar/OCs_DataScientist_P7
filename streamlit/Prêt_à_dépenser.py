import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Prêt à dépenser! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ### This dashboard was created to provide information about customers registered in our database. 
    - On the first page you can search for a **customer's probability of paying a loan** based on their ID, as well as some information about them. 
    - On the second page you can obtain **information about customers similar to the researched customer**. 
"""
)