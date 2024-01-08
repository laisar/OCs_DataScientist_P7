import streamlit as st
import requests
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import shap
import numpy as np
import os


########################################################
# Loading images to the website
########################################################
#icon = Image.open("favicon.ico")
path1 = os.path.dirname(__file__)
path = os.getcwd()
logo = Image.open(path1+"/images/logo.png")
finalpath = path1+"/images/logo.png"

########################################################
# General settings
########################################################
st.set_page_config(
    page_title="Prêt à dépenser - Default Risk",
    page_icon="💶",
    layout="wide",
    initial_sidebar_state="expanded"
)
def list_folders(directory):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    return folders
folders_in_directory = list_folders(path1)

st.write(str(path1))
st.write(str(path))
st.write(str(finalpath))

print("Folders in", path1, "are:")
for folder in folders_in_directory:
    st.write(folder)
    st.write("\n")
files = os.listdir(path1)

# Print the list of files
for file in files:
    st.write(file)
########################################################
# Page information
########################################################
st_title = '<h1 style="color:#262730; margin-bottom:0; padding: 1.25rem 0px 0rem;">Prêt à dépenser - Default Risk</h1>'
st_title_hr = '<hr style="background-color:#F0F2F6; width:60%; text-align:left; margin-left:0; margin-top:0">'
st.markdown(st_title, unsafe_allow_html=True)
st.markdown(st_title_hr, unsafe_allow_html=True)


API_ADDRESS = 'https://fastapilaisar.azurewebsites.net'

url_current = "https://github.com/laisar/OCs_DataScientist_P7/blob/master/app2/dataset_target_streamlit_compressed.gz?raw=true"
df_clients_target = pd.read_csv(path1+"/dataset_target_streamlit_compressed.gz", compression='gzip', sep=',')


def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

mygrid = make_grid(2,1)

with mygrid[0][0]:

    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image(logo, width=200) 
        st.write("test") 

with mygrid[1][0]:

    st.subheader("Implement a scoring model - - OpenClassrooms Data Scientist P7")

    st.markdown(""" <h5 style="text-align: left;">✍️ Made by</h5> """, unsafe_allow_html=True)

    made_by = '<ul style="list-style-type:disc;">'\
                        '<li>Made by <a href="https://www.linkedin.com/in/lais-amorim-reis/" target="_blank">Lais Amorim Reis</a>.</li>'\
                    '</ul>'
    st.markdown(made_by, unsafe_allow_html=True)
    
    st.markdown(""" <h5 style="text-align: left;">🖥️ For more information</h5> """, unsafe_allow_html=True)

    visit_github = '<ul style="list-style-type:disc;">'\
                        '<li>Visit the <a href="https://github.com/laisar/OCs_DataScientist_P7" target="_blank">GitHub</a> repository.</li>'\
                    '</ul>'
    st.markdown(visit_github, unsafe_allow_html=True)

