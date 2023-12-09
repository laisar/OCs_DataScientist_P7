import streamlit as st
import matplotlib.pyplot as plt
from streamlit_searchbox import st_searchbox
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Rectangle
import time
import numpy as np
import requests
import re
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import shap
import os
import requests
import json
import urllib.request
import joblib
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
#Options
st.set_option('deprecation.showPyplotGlobalUse', False)

#Nom de la page
st.set_page_config(
    page_title= "Global analysis", 
    page_icon= "📊",
    layout="wide"
    )

# Suppression des marges par défaut
padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

#Titre
html_header="""
    <head> 
    <center>
        <title>Application Dashboard Crédit Score - Analyse Client</title> <center>
        <meta charset="utf-8">
        <meta name="description" content="Analyse client">
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>             
    <h1 style="font-size:300%; color:Crimson; font-family:Arial"> Prêt à dépenser <br>
        <h2 style="color:Gray; font-family:Georgia"> Global dashboard</h2>
        <hr style= "  display: block;
          margin-top: 0;
          margin-bottom: 0;
          margin-left: auto;
          margin-right: auto;
          border-style: inset;
          border-width: 1.5px;"/>
     </h1>
"""

st.markdown('<style>body{background-color: #fbfff0}</style>',unsafe_allow_html=True)
st.markdown(html_header, unsafe_allow_html=True)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    # Columns to exclude
    columns_to_exclude = ["Client ID", "Gender", "Married", "Working", "Owns a car", "Owns a real estate property"]

    # Create a list of DataFrame columns excluding specified columns
    filtered_columns = df.columns.difference(columns_to_exclude).tolist()
    
    for col in filtered_columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", filtered_columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def plot_pie_plot(data, column):
    
    df = data[column]
    
    # Get unique values and their frequencies
    unique_values = data[column].value_counts().index.tolist()
    frequencies = data[column].value_counts().values.tolist()

    # Create a pie chart using Plotly Express
    fig = px.pie(names=unique_values, values=frequencies, title=column)
    
    fig.update_layout(width=400, height=400)
    # Show the plot
    st.plotly_chart(fig)
    
def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid
url_current = "https://github.com/laisar/OCs_DataScientist_P7/blob/master/dataset_predict_compressed.gz?raw=true"
df_current_clients = pd.read_csv(url_current, compression='gzip', sep=',')
df_current_clients = df_current_clients.drop(columns=["SK_ID_CURR", "TARGET", "REPAY", "CLUSTER"])
path = os.path.dirname(__file__)
my_file_shap = path+'/shap_explainer.pckl'
explainer = joblib.load(my_file_shap)
shap_values = explainer.shap_values(df_current_clients)

st.write("") 

# Create tabs
tab_global, tab_dependecy, tab_clients = st.tabs(["Global interpretation", "Dependency graph", "Similar clients"])
    
# Make an API request to the FastAPI backend
response = requests.get("http://54.198.181.125/api/clients")  # Replace with the actual URL of your FastAPI server
clients = response.json()
my_list = clients["clientsId"]

# Make an API request to the FastAPI backend
choice = True
response_repay = requests.get("http://54.198.181.125/api/predictions/list_clients?id=true")  # Replace with the actual URL of your FastAPI server
clients_repay = response_repay.json()

# Make an API request to the FastAPI backend
response_dontrepay = requests.get("http://54.198.181.125/api/predictions/list_clients?id=false")  # Replace with the actual URL of your FastAPI server
clients_dontrepay = response_dontrepay.json()


selected_option = st.sidebar.radio("Customers:", ["All customers", "Allow loan", "Do not allow loan"], index=0)

if selected_option == "All customers":
    selected_client = st.sidebar.selectbox("Select a customer: ", my_list)
if selected_option == "Allow loan":
    selected_client = st.sidebar.selectbox("Select a customer: ", clients_repay)
if selected_option == "Do not allow loan":
    selected_client = st.sidebar.selectbox("Select a customer: ", clients_dontrepay)
    
with tab_global:
    fig = plt.figure(figsize=(25, 25))
    plt.title("Global interpretation :\n Impact of each feature on prediction\n")
    st_shap(shap.summary_plot(shap_values[1], 
                             features=df_current_clients,
                             feature_names=df_current_clients.columns,
                             plot_size=(12, 16),
                             cmap='PiYG_r',
                             plot_type="dot",
                             max_display=56,
                             show = False))
    plt.show()

with tab_dependecy:
    st.markdown("""
             <h1 style="color:772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
             Dependency graph</h1>
             """, 
             unsafe_allow_html=True)
    st.write("We can get a deeper insight into the effect of each featur on the dataset with a dependency graph.")
    st.write("The dependence plot allows variables to be analyzed in pairs, suggesting the possibility of observing interactions. The scatter plot represents a dependency between a variable (in x) and the shapley values (in y) colored by the most correlated variable.")


    # Création et affichage du sélecteur des variables et des graphs de dépendance 

    liste_variables = df_current_clients.columns.to_list()

    col1, col2, = st.columns(2)  #division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
    with col1:
        ID_var = st.selectbox("*Please select a variable from the drop-down menu 👇*", 
                                 (liste_variables))
        st.write("You have selected the variable :", ID_var)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    shap.dependence_plot(ID_var, 
                     shap_values[1], 
                     df_current_clients, 
                     interaction_index=None,
                     alpha = 0.5,
                     x_jitter = 0.5,
                     title= "Dependency graph",
                     ax=ax1,
                     show = False)
    ax2 = fig.add_subplot(122)
    shap.dependence_plot(ID_var, 
                     shap_values[1], 
                     df_current_clients, 
                     interaction_index='auto',
                     alpha = 0.5,
                     x_jitter = 0.5,
                     title= "Dependency and Interaction Graphs",
                     ax=ax2,
                     show = False)
    fig.tight_layout()
    st.pyplot(fig)
    
with tab_clients:
    
    columns_to_drop = ["Gender", "Married", "Working", "Owns a car", "Owns a real estate property"]
    
    similar_clients = requests.get(f"http://54.198.181.125/api/clients/similar_clients?id={selected_client}")
    similar_clients = similar_clients.json()
    client_info = requests.get(f"http://54.198.181.125/api/clients/clients_info/?id={selected_client}")
    client_info = client_info.json()
    client_info = pd.DataFrame(client_info, index=['0'])
    fields = client_info.columns.tolist()
    list_clients = []
    
    for client in similar_clients:
        client_info = requests.get(f"http://54.198.181.125/api/clients/clients_info/?id={client}")
        client_info = client_info.json()
        client_info = pd.DataFrame(client_info, index=['0'])
        data_fields = client_info.iloc[0].tolist()
        list_clients.append(data_fields)
        
    data = pd.DataFrame(list_clients, columns=fields)
    data = filter_dataframe(data)
    
    mean_values = data.select_dtypes(include='number')
    mean_values = mean_values.mean()
    mean_df = pd.DataFrame(mean_values, columns=['Mean']).transpose()
    
    data_show = data.drop(columns = columns_to_drop)
    
    st.dataframe(data_show.set_index(data_show.columns[0]))
    st.write("")
    st.dataframe(mean_df.drop(columns=["Client ID"]))
    st.write("")
    
    mygrid = make_grid(2,1)
    
    with mygrid[0][0]:
        
        Gender, Married, Working = st.columns(3)
        
        with Gender:
            plot_pie_plot(data, "Gender") 

        with Married:
            plot_pie_plot(data, "Married")

        with Working:
            plot_pie_plot(data, "Working")
        
    with mygrid[1][0]:
    
        Car, Property  = st.columns(2)
    
        with Car:
            plot_pie_plot(data, "Owns a car")

        with Property:
            plot_pie_plot(data, "Owns a real estate property")
    
    #st.markdown(data.drop(columns = colmuns_to_drop).style.hide(axis="index").to_html(), unsafe_allow_html=True)
