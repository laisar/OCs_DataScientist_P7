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

########################################################
# General settings
########################################################
st.set_page_config(
    page_title="Prêt à dépenser - Default Risk: Local Analysis",
    page_icon="💶",
    layout="wide",
    initial_sidebar_state="expanded"
)

########################################################
# Page information
########################################################
st_title = '<h1 style="color:#262730; margin-bottom:0; padding: 1.25rem 0px 0rem;">Prêt à dépenser - Default Risk: Local Analysis</h1>'
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

def get_client_shap_values(id):
    response = requests.get(API_ADDRESS + f'api/clients/prediction/shap/local?id={id}')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

def get_model_shap_values(id):
    response = requests.get(API_ADDRESS + f'/api/clients/prediction/shap/global?id={id}')
    if response.status_code == 200: 
        data = response.json()
        return data
    else:
        st.error('Failed to get clients')
        return None

def plot_shap(data: dict):

    df = pd.DataFrame({'Features': data.keys(), 'Importance': data.values()})
    df['Color'] = df['Importance'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')

    colors = {'Positive': 'limegreen', 'Negative': 'orangered'}

    fig = px.bar(df, x='Importance', y='Features', color='Color', color_discrete_map=colors)
    fig.update_layout(barmode='group', xaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig, use_container_width=True)

tab1, tab2, tab3 = st.tabs(['👥 Client information', 
                            '🎯 Client score', 
                            '📊 Local feature importance'
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

client_probability = requests.get(f"https://fastapilaisar.azurewebsites.net/api/predictions/clients/?id={selected_client}")
client_probability = client_probability.json()

client_info = requests.get(f"https://fastapilaisar.azurewebsites.net/api/clients/clients_info/?id={selected_client}")
client_info = client_info.json()

with tab1:

    st.markdown(""" <h4 style="text-align: center;">General client information</h4> """, unsafe_allow_html=True)
    st.markdown('\n')

    client_info = pd.DataFrame(client_info, index=['0'])
    fields = client_info.columns.tolist()
    data_fields = client_info.iloc[0].tolist()
    data = pd.DataFrame(list(zip(fields, data_fields)), columns=[str(selected_client), "Client information"])
    mygrid = make_grid(3,1)
    path = 'images'
    with mygrid[0][0]:
        
        mygrid2 = make_grid(1,3)
        
        with mygrid2[0][0]:
            # id
            st.markdown("**Client's ID:**")
            st.write(str(selected_client))
                
        with mygrid2[0][1]:
            # age
            st.markdown('**Age:**')
            st.write(f"Is {data[data[str(selected_client)] == 'Age']['Client information'].values[0]} years old")
                
        with mygrid2[0][2]:
            # gender
            st.markdown('**Gender:**')
            if((data.loc[data[str(selected_client)] == "Gender", "Client information"] == "Woman").any()):             
                my_file = path+'/woman.png'
                image = Image.open(my_file)
                st.image(image, use_column_width=False, caption='Woman')
            else:
                my_file = path+'/man.png'
                image = Image.open(my_file)
                st.image(image, use_column_width=False, caption='Man')

    with mygrid[1][0]:
        
        mygrid3 = make_grid(1,3)
        
        with mygrid3[0][0]:
            #married
            st.markdown('**Marital status:**')
            if((data.loc[data[str(selected_client)] == "Married", "Client information"] == "Yes").any()):             
                my_file = path+'/rings.png'
                image = Image.open(my_file)
                st.image(image, use_column_width=False, caption='Is married')
            else:
                my_file = path+'/rings.png'
                image = Image.open(my_file)
                st.image(image, use_column_width=False, caption='Is not married')
                
        with mygrid3[0][1]:
            #children
            st.markdown('**Has children:**')
            my_file = path+'/child.png'
            image = Image.open(my_file)
            if((data.loc[data[str(selected_client)] == "Number of children", "Client information"] >=1).any()):
                st.image(image, use_column_width=False, caption=f"Has {data[data[str(selected_client)] == 'Number of children']['Client information'].values[0]} children")    
            else:
                image = image.convert('L')
                st.image(image, use_column_width=False, caption='Does not have kids')  
                
        with mygrid3[0][2]:
            #job
            st.markdown('**Working status:**')
            my_file = path+'/working.png'
            image = Image.open(my_file)
            if((data.loc[data[str(selected_client)] == "Working", "Client information"] == "Yes").any()):
                st.image(image, use_column_width=False, caption='Has a job')
                
            else:
                image = image.convert('L')
                st.image(image, use_column_width=False, caption='Does not have a job') 

    with mygrid[2][0]:
        
        mygrid4 = make_grid(1,3)
        
        with mygrid4[0][0]:
            #rent
            st.markdown('**Owns a real estate property:**')
            my_file_house = path+'/house.png'
            image_house = Image.open(my_file_house)
            if((data.loc[data[str(selected_client)] == "Owns a real estate property", "Client information"] == "Yes").any()):
                st.image(image_house, use_column_width=False, caption='Yes')
                
            else:
                image_house = image_house.convert('L')
                st.image(image_house, use_column_width=False, caption='No') 
                
        with mygrid4[0][1]:
            #car
            st.markdown('**Owns a vehicle:**')
            my_file_car = path+'/car.png'
            image_car = Image.open(my_file_car)
            if((data.loc[data[str(selected_client)] == "Owns a car", "Client information"] == "Yes").any()):
                st.image(image_car, use_column_width=False, caption='Yes')
                
            else:
                image_car = image_car.convert('L')
                st.image(image_car, use_column_width=False, caption='No')   
                
        with mygrid4[0][2]:
            #working since
            st.markdown('**Working since:**')
            st.write(f"Working since {data[data[str(selected_client)] == 'Working since']['Client information'].values[0]}")

    st.markdown(""" <h4 style="text-align: center;">General financial information</h4> """, unsafe_allow_html=True)
    st.markdown('\n')

    mygrid5 = make_grid(2,1)
    
    with mygrid5[0][0]:

        mygrid6 = make_grid(1,3)

        with mygrid6[0][0]:

            st.markdown('**Total credit amount:**')
            st.write(f"$ {data[data[str(selected_client)] == 'Total credit amount']['Client information'].values[0]}")

        with mygrid6[0][1]:

            st.markdown('**Credit amount repaied per year:**')
            st.write(f"$ {data[data[str(selected_client)] == 'Credit amount repaied per year']['Client information'].values[0]}")

        with mygrid6[0][2]:

            st.markdown('**Client anual income:**')
            st.write(f"$ {data[data[str(selected_client)] == 'Client anual income']['Client information'].values[0]}")

    with mygrid5[1][0]:

        mygrid7 = make_grid(1,3)

        with mygrid7[0][0]:

            st.markdown("**Client's payment rate:**")
            st.write(f"{data[data[str(selected_client)] == 'Payment rate (%)']['Client information'].values[0]} %")

        with mygrid7[0][1]:

            st.markdown('**Source 2:**')
            st.write(f"{data[data[str(selected_client)] == 'Source 2 (%)']['Client information'].values[0]} %")

        with mygrid7[0][2]:

            st.markdown('**Source 3:**')
            st.write(f"{data[data[str(selected_client)] == 'Source 3 (%)']['Client information'].values[0]} %")


with tab2:

    mygrid = make_grid(1,2)

    with mygrid[0][0]:

        if 0 <= client_probability["probability0"] < 0.2:
            result = 1
            phrase = "The client's score is much lower than the score for loan authorization"
        elif 0.2 <= client_probability["probability0"] < 0.4:
            result = 2
            phrase = "The client's score is lower than the score for loan authorization"

        elif 0.4 <= client_probability["probability0"] < 0.581:
            result = 3
            phrase = "The client's score is considered fair, the loan should not be authorized"

        elif 0.581 <= client_probability["probability0"] < 0.8:
            result = 4
            phrase = "The client's score is good, the loan can be authorized"

        elif 0.8 <= client_probability["probability0"] <= 1:
            result = 5
            phrase = "The client's score is excellent, the loan can be authorized"

        gauge(labels=['VERY POOR', 'POOR', 'FAIR', 'GOOD', 'EXCELLENT'], \
                colors=["#f20505", "#f28b05", '#eef205', '#58f723', '#00a326'], arrow=result, title=round(client_probability["probability0"]*1000,0))

        st.set_option('deprecation.showPyplotGlobalUse', False)    
        st.pyplot()

    with mygrid[0][1]:

        st.markdown('**Score explanation:**')
        # Write text
        st.write("The threshold that determines whether credit should be agreed is 0.419, i.e. customers whose probability of defaulting is higher than 41.9% should not have their credit approved")
        st.write("- Probability of repaying a loan: " + str(round(client_probability["probability0"]*100,2)) + "%")
        st.write("- Probability of not repaying a loan: " + str(round(client_probability["probability1"]*100,2)) + "%")  

with tab3:

    st.markdown('**See local feature importance below:**')

    client_values = get_client_shap_values(selected_client)
    plot_shap(client_values)