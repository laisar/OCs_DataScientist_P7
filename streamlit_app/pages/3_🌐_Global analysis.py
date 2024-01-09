import streamlit as st
import requests
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Wedge, Rectangle
import pandas as pd
import joblib
import shap
import numpy as np
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

########################################################
# General settings
########################################################
st.set_page_config(
    page_title="Prêt à dépenser - Default Risk: Global Analysis",
    page_icon="💶",
    layout="wide",
    initial_sidebar_state="expanded"
)

########################################################
# Page information
########################################################
st_title = '<h1 style="color:#262730; margin-bottom:0; padding: 1.25rem 0px 0rem;">Prêt à dépenser - Default Risk: Global Analysis</h1>'
st_title_hr = '<hr style="background-color:#F0F2F6; width:60%; text-align:left; margin-left:0; margin-top:0">'
st.markdown(st_title, unsafe_allow_html=True)
st.markdown(st_title_hr, unsafe_allow_html=True)

API_ADDRESS = 'https://fastapilaisar.azurewebsites.net/'

def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

def get_model_shap_values(id):
    response = requests.get(API_ADDRESS + f'api/clients/prediction/shap/global?id={id}')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

def get_similar_clientes(id, kclients):
    response = requests.get(API_ADDRESS + f'api/clients/similar_clients?id={id}&k={kclients}')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

def plot_shap(data: dict):

    df = pd.DataFrame({'Features': data.keys(), 'Importance': data.values()})
    df['Color'] = df['Importance'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    df = df.sort_values(by='Importance', ascending=True)

    colors = {'Positive': 'limegreen', 'Negative': 'orangered'}

    fig = px.bar(df, x='Importance', y='Features', color='Color', color_discrete_map=colors)
    fig.update_layout(height=800, width=800) 
    st.plotly_chart(fig, use_container_width=True)

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

tab1, tab2 = st.tabs(['👥 Similar clients', 
                            '📊 Global feature importance'
                        ])

# Make an API request to the FastAPI backend
response = requests.get(API_ADDRESS+"api/clients")  # Replace with the actual URL of your FastAPI server
clients = response.json()
my_list = clients["clientsId"]

# Make an API request to the FastAPI backend
response_repay = requests.get(API_ADDRESS+"api/predictions/list_clients?id=true")  # Replace with the actual URL of your FastAPI server
clients_repay = response_repay.json()

# Make an API request to the FastAPI backend
response_default = requests.get(API_ADDRESS+"api/predictions/list_clients?id=false")  # Replace with the actual URL of your FastAPI server
response_default = response_default.json()

selected_option = st.sidebar.radio("Filter clients:", ["All clients", "Allow loan", "Reject loan"], index=0)

if selected_option == "All clients":
    selected_client = st.sidebar.selectbox("Select a client: ", my_list)
if selected_option == "Allow loan":
    selected_client = st.sidebar.selectbox("Select a client: ", clients_repay)
if selected_option == "Reject loan":
    selected_client = st.sidebar.selectbox("Select a client: ", response_default)

with tab1:

    st.markdown(""" <h5 style="text-align: center;">See below the dataframe of the clients that are similar to the selected client</h5> """, unsafe_allow_html=True)
    st.markdown('\n')

    clients_filter = st.slider('Select the number of  clients', 2, 100, 20)
    df_clients = get_similar_clientes(selected_client, clients_filter)

    str_list = [str(i) for i in range(len(df_clients))]
    df_clients = pd.DataFrame(df_clients, index=str_list)

    df_clients['SK_ID_CURR'] = df_clients['SK_ID_CURR'].astype(str)
    df_clients['CODE_GENDER'] = df_clients['CODE_GENDER'].replace({0: "Man", 1: "Woman"})
    df_clients['REPAY'] = df_clients['REPAY'].replace({False: "Default", True: "Repay"})
    df_clients['NAME_FAMILY_STATUS_Married'] = df_clients['NAME_FAMILY_STATUS_Married'].replace({False: "Not married", True: "Married"})
    df_clients['NAME_INCOME_TYPE_Working'] = df_clients['NAME_INCOME_TYPE_Working'].replace({True: "Has a job", False: "Doesn't have a job"})
    df_clients['FLAG_OWN_CAR'] = df_clients['FLAG_OWN_CAR'].replace({1: "Has a vehicle", 0: "Doesn't have a vehicle"})
    df_clients['FLAG_OWN_REALTY'] = df_clients['FLAG_OWN_REALTY'].replace({1: "Has a property", 0: "Doesn't have a property"})

    df_filtered = filter_dataframe(df_clients)

    st.write(df_filtered)

    mygrid = make_grid(3, 2)

    with mygrid[0][0]:
        st.markdown('**Loan repayment distribution:**')
        # Plotting with Plotly Express
        df = df_filtered.groupby(['REPAY'])['REPAY'].count().reset_index(name='count')
        fig = px.pie(df, values='count', names='REPAY')
        # Displaying the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with mygrid[0][1]:
        st.markdown('**Gender distribution:**')
        # Plotting with Plotly Express
        df = df_filtered.groupby(['CODE_GENDER'])['CODE_GENDER'].count().reset_index(name='count')
        fig = px.pie(df, values='count', names='CODE_GENDER')
        # Displaying the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with mygrid[1][0]:
        st.markdown('**Anual income distribution:**')
        # Create histogram using Plotly Express
        df1 = df_filtered[df_filtered["REPAY"] == "Default"].reset_index(drop=True)
        df2 = df_filtered[df_filtered["REPAY"] == "Repay"].reset_index(drop=True)
        fig1 = px.histogram(df1, x='AMT_INCOME_TOTAL', color_discrete_sequence=['orangered'])
        fig2 = px.histogram(df2, x='AMT_INCOME_TOTAL', color_discrete_sequence=['limegreen'])
        #fig = px.histogram(df_filtered, x='AMT_INCOME_TOTAL')
        tab3, tab4 = st.tabs(['Prediction: Repay', 
                            'Prediction: Default'
                        ])
        with tab3:
        # Display the histogram using Streamlit
            st.plotly_chart(fig2, use_container_width=True)
        with tab4:
            st.plotly_chart(fig1, use_container_width=True)


    with mygrid[1][1]:
        st.markdown('**Total credit amount distribution:**')
        # Create histogram using Plotly Express
        df1 = df_filtered[df_filtered["REPAY"] == "Default"].reset_index(drop=True)
        df2 = df_filtered[df_filtered["REPAY"] == "Repay"].reset_index(drop=True)
        fig1 = px.histogram(df1, x='AMT_CREDIT', color_discrete_sequence=['orangered'])
        fig2 = px.histogram(df2, x='AMT_CREDIT', color_discrete_sequence=['limegreen'])
        #fig = px.histogram(df_filtered, x='AMT_INCOME_TOTAL')
        tab5, tab6 = st.tabs(['Prediction: Repay', 
                            'Prediction: Default'
                        ])
        with tab5:
        # Display the histogram using Streamlit
            st.plotly_chart(fig2, use_container_width=True)
        with tab6:
            st.plotly_chart(fig1, use_container_width=True)

    with mygrid[2][0]:
        st.markdown('**Credit amount repaied per year:**')
        # Create histogram using Plotly Express
        df1 = df_filtered[df_filtered["REPAY"] == "Default"].reset_index(drop=True)
        df2 = df_filtered[df_filtered["REPAY"] == "Repay"].reset_index(drop=True)
        fig1 = px.histogram(df1, x='AMT_ANNUITY', color_discrete_sequence=['orangered'])
        fig2 = px.histogram(df2, x='AMT_ANNUITY', color_discrete_sequence=['limegreen'])
        #fig = px.histogram(df_filtered, x='AMT_INCOME_TOTAL')
        tab7, tab8 = st.tabs(['Prediction: Repay', 
                            'Prediction: Default'
                        ])
        with tab7:
        # Display the histogram using Streamlit
            st.plotly_chart(fig2, use_container_width=True)
        with tab8:
            st.plotly_chart(fig1, use_container_width=True)


    with mygrid[2][1]:
        st.markdown('**Average normalized scores from external data sources:**')
        # Create histogram using Plotly Express
        df1 = df_filtered[df_filtered["REPAY"] == "Default"].reset_index(drop=True)
        df2 = df_filtered[df_filtered["REPAY"] == "Repay"].reset_index(drop=True)

        tab9, tab10 = st.tabs(['Prediction: Repay', 
                            'Prediction: Default'
                        ])
        with tab9:
        # Display the histogram using Streamlit
            values1 = round(df2["EXT_SOURCE_2"].mean(),2)
            values2 = round(df2["EXT_SOURCE_3"].mean(),2)
            df_bar = {
                'EXT_SOURCE_MEAN': ['EXT_SOURCE_2', 'EXT_SOURCE_3'],
                'Values': [values1, values2]
            }
            colors = ['limegreen', 'limegreen']
            fig = px.bar(df_bar, x='EXT_SOURCE_MEAN', y='Values', color='EXT_SOURCE_MEAN', color_discrete_sequence=colors)
            fig.update_layout(
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab10:
            values1 = round(df1["EXT_SOURCE_2"].mean(),2)
            values2 = round(df1["EXT_SOURCE_3"].mean(),2)
            df_bar = {
                'EXT_SOURCE_MEAN': ['EXT_SOURCE_2', 'EXT_SOURCE_3'],
                'Values': [values1, values2]
            }
            colors = ['orangered', 'orangered']
            fig = px.bar(df_bar, x='EXT_SOURCE_MEAN', y='Values', color='EXT_SOURCE_MEAN', color_discrete_sequence=colors)
            fig.update_layout(
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
    
with tab2:

    st.markdown('**See global feature importance below:**')

    features_filter = st.slider('Select the number of features', 0, 45, 15)

    client_values = get_model_shap_values(features_filter)

    plot_shap(client_values)
