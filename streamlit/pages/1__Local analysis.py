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
from PIL import Image
import shap
import requests
import json
import urllib.request
import joblib
#Options
st.set_option('deprecation.showPyplotGlobalUse', False)

#Nom de la page
st.set_page_config(
    page_title= "Local analysis", 
    page_icon= "🔎",
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
        <h2 style="color:Gray; font-family:Georgia"> Local dashboard</h2>
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
#html_header

def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid
#st.set_page_config(page_title="Customer's score and info", page_icon="📈")

#st.markdown("## Select a customer's ID to see their probability of repaying a loan")
#st.sidebar.header("Customer's score and info")
st.write("") 

# Create tabs
tab_score, tab_information, tab_feature_importance = st.tabs(["Client score", "Client information", "Client feature importance"])

# Make an API request to the FastAPI backend
response = requests.get("http://54.198.181.125/api/clients")  # Replace with the actual URL of your FastAPI server
clients = response.json()
my_list = clients["clientsId"]

explainer = pickle.load(open('../models/shap_explainer.pckl', 'rb'))

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

client_probability = requests.get(f"http://54.198.181.125/api/predictions/clients/?id={selected_client}")
client_probability = client_probability.json()

client_info = requests.get(f"http://54.198.181.125/api/clients/clients_info/?id={selected_client}")
client_info = client_info.json()

client_shap = requests.get(f"http://54.198.181.125/api/clients/client?id={selected_client}")
client_shap = client_shap.json()
client_shap = pd.DataFrame(client_shap) 

shap_values = explainer.shap_values(client_shap)

#client_shap = requests.get(f"http://54.198.181.125/api/clients/shap/?id={selected_client}")
#client_shap = client_shap.json()

with tab_score:
    # Write text
    st.write("- Probability of repaying a loan: " + str(round(client_probability["probability0"]*100,2)) + "%")
    st.write("- Probability of not repaying a loan: " + str(round(client_probability["probability1"]*100,2)) + "%")  
    #st.write(client_info)

    if 0 <= client_probability["probability0"] < 0.2:
        result = 1
        phrase = "The client's score is much lower than the score for loan authorization"
    elif 0.2 <= client_probability["probability0"] < 0.4:
        result = 2
        phrase = "The client's score is lower than the score for loan authorization"

    elif 0.4 <= client_probability["probability0"] < 0.633:
        result = 3
        phrase = "The client's score is considered fair, but it is recommended that the loan not be authorized"

    elif 0.633 <= client_probability["probability0"] < 0.8:
        result = 4
        phrase = "The customer's score is good, the loan can be authorized"

    elif 0.8 <= client_probability["probability0"] <= 1:
        result = 5
        phrase = "The customer's score is excellent, the loan can be authorized"


    def degree_range(n):
        start = np.linspace(0, 180, n + 1, endpoint=True)[0:-1]
        end = np.linspace(0, 180, n + 1, endpoint=True)[1::]
        mid_points = start + ((end - start) / 2.)
        return np.c_[start, end], mid_points


    def rot_text(ang):
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation


    def gauge(labels=['VERY POOR', 'POOR', 'FAIR', 'GOOD', 'EXCELLENT'], \
              colors='jet_r', arrow=1, title='', fname=False):


        N = len(labels)

        if arrow > N:
            raise Exception("\n\nThe category ({}) is greated than \
            the length\nof the labels ({})".format(arrow, N))


        if isinstance(colors, str):
            cmap = cm.get_cmap(colors, N)
            cmap = cmap(np.arange(N))
            colors = cmap[::-1, :].tolist()
        if isinstance(colors, list):
            if len(colors) == N:
                colors = colors[::-1]
            else:
                raise Exception("\n\nnumber of colors {} not equal \
                to number of categories{}\n".format(len(colors), N))


        fig, ax = plt.subplots()

        ang_range, mid_points = degree_range(N)

        labels = labels[::-1]

        patches = []
        for ang, c in zip(ang_range, colors):
            # sectors
            patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
            # arcs
            patches.append(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

        foo = [ax.add_patch(p) for p in patches]


        for mid, lab in zip(mid_points, labels):
            ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
                    horizontalalignment='center', verticalalignment='center', fontsize=14, \
                    fontweight='bold', rotation=rot_text(mid))

        r = Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2)
        ax.add_patch(r)

        ax.text(0, -0.05, title, horizontalalignment='center', \
                verticalalignment='center', fontsize=12, fontweight='bold')

        pos = mid_points[abs(arrow - N)]

        ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))
        plt.title(phrase, fontsize = 10, fontweight='bold')
        ax.set_frame_on(False)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()
        if fname:
            fig.savefig(fname, dpi=200)

    gauge(labels=['VERY POOR', 'POOR', 'FAIR', 'GOOD', 'EXCELLENT'], \
            colors=["#f20505", "#f28b05", '#eef205', '#58f723', '#00a326'], arrow=result, title=round(client_probability["probability0"]*1000,0))

    st.set_option('deprecation.showPyplotGlobalUse', False)    
    st.pyplot()
    
with tab_information:
    
    client_info = pd.DataFrame(client_info, index=['0'])
    fields = client_info.columns.tolist()
    data_fields = client_info.iloc[0].tolist()
    data = pd.DataFrame(list(zip(fields, data_fields)), columns=[str(selected_client), "Client information"])
    mygrid = make_grid(2,1)
    
    with mygrid[0][0]:
        
        mygrid2 = make_grid(1,3)
        
        with mygrid2[0][0]:
            st.write(str(selected_client))
            if((data.loc[data[str(selected_client)] == "Gender", "Client information"] == "Woman").any()):
                image_path = "../streamlit/images/woman.png"
                image = Image.open(image_path)
                st.image(image, use_column_width=False, caption='Woman')
            else:
                image_path = "../streamlit/images/man.png"
                image = Image.open(image_path)
                st.image(image, use_column_width=False, caption='Man')
                
        with mygrid2[0][1]:
            st.write("Owns a car")
            image_path_car = "../streamlit/images/car.png"
            image_car = Image.open(image_path_car)
            if((data.loc[data[str(selected_client)] == "Owns a car", "Client information"] == "Yes").any()):
                st.image(image_car, use_column_width=False, caption='Yes')
            else:
                image_car = image_car.convert('L')
                st.image(image_car, use_column_width=False, caption='No')  
                
        with mygrid2[0][2]:
            st.write("Owns a real estate property")
            image_path_prop = "../streamlit/images/house.png"
            image_prop = Image.open(image_path_prop)
            if((data.loc[data[str(selected_client)] == "Owns a real estate property", "Client information"] == "Yes").any()):
                st.image(image_prop, use_column_width=False, caption='Yes')
            else:    
                image_prop = image_prop.convert('L')
                st.image(image_prop, use_column_width=False, caption='No') 

    with mygrid[1][0]:
        
        st.dataframe(data.set_index(data.columns[0]))
      
with tab_feature_importance:
    st.pyplot(shap.plots.force(explainer.expected_value[1], shap_values[1], client_shap, matplotlib=True))

    shap.decision_plot(explainer.expected_value[1], shap_values[1], client_shap)

    st.pyplot()
    #st.write(client_shap)
