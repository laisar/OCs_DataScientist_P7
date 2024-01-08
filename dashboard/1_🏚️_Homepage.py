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
path = os.path.dirname(__file__)
logo = Image.open(path+"/logo.png")


########################################################
# General settings
########################################################
st.set_page_config(
    page_title="Prêt à dépenser - Default Risk",
    page_icon="💶",
    layout="wide",
    initial_sidebar_state="expanded"
)

########################################################
# Page information
########################################################
st_title = '<h1 style="color:#262730; margin-bottom:0; padding: 1.25rem 0px 0rem;">Prêt à dépenser - Default Risk</h1>'
st_title_hr = '<hr style="background-color:#F0F2F6; width:60%; text-align:left; margin-left:0; margin-top:0">'
st.markdown(st_title, unsafe_allow_html=True)
st.markdown(st_title_hr, unsafe_allow_html=True)


API_ADDRESS = 'https://fastapilaisar.azurewebsites.net'

url_current = "https://github.com/laisar/OCs_DataScientist_P7/blob/master/dashboard/dataset_target_compressed.gz?raw=true"
df_current_clients = pd.read_csv(url_current, compression='gzip', sep=',')


def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

@st.cache_data 
def get_gender():
    response = requests.get(API_ADDRESS + '/api/clients/gender')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data 
def plot_gender(data: dict):

    image1 = Image.open(path+"/woman.png")
    image2 = Image.open(path+"/man.png")

    # Create a Matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 7))

    # Plot the first image on the left subplot
    ax1.imshow(np.array(image2))
    ax1.text(0.5, -0.1, f"Men: {list(data.values())[0]:.2f}%", size=8, ha="center", transform=ax1.transAxes)

    # Plot the second image on the right subplot
    ax2.imshow(np.array(image1))
    ax2.text(0.5, -0.1, f"Women: {list(data.values())[1]:.2f}%", size=8, ha="center", transform=ax2.transAxes)

    # Remove axis labels
    ax1.axis('off')
    ax2.axis('off')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

@st.cache_data 
def get_house():
    response = requests.get(API_ADDRESS + '/api/clients/house')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data 
def plot_house(data: dict):

    image1 = Image.open(path+"/house.png")
    image2 = Image.open(path+"/rent.png")

    # Create a Matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))

    # Plot the first image on the left subplot
    ax1.imshow(np.array(image1))
    ax1.text(0.5, -0.1, f"Owns a property: {list(data.values())[0]:.2f}%", size=8, ha="center", transform=ax1.transAxes)

    fig.subplots_adjust(wspace=0.3)

    # Plot the second image on the right subplot
    ax2.imshow(np.array(image2))
    ax2.text(0.5, -0.1, f"Doesn't own a property: {list(data.values())[1]:.2f}%", size=8, ha="center", transform=ax2.transAxes)

    # Remove axis labels
    ax1.axis('off')
    ax2.axis('off')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

@st.cache_data 
def get_car():
    response = requests.get(API_ADDRESS + '/api/clients/car')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data 
def plot_car(data: dict):

    image1 = Image.open(path+"/car.png")
    image2 = Image.open(path+"/walk.png")

    # Create a Matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))

    # Plot the first image on the left subplot
    ax1.imshow(np.array(image1))
    ax1.text(0.5, -0.1, f"Owns a car: {list(data.values())[0]:.2f}%", size=8, ha="center", transform=ax1.transAxes)

    # Plot the second image on the right subplot
    ax2.imshow(np.array(image2))
    ax2.text(0.5, -0.1, f"Doesn't own a car: {list(data.values())[1]:.2f}%", size=8, ha="center", transform=ax2.transAxes)

    # Remove axis labels
    ax1.axis('off')
    ax2.axis('off')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

@st.cache_data 
def get_repay():
    response = requests.get(API_ADDRESS + '/api/clients/repay')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None

def create_histogram_credit():

    data = df_clients_target[["TARGET", "AMT_CREDIT"]]

    # Condition to segment the DataFrame 
    condition1 = data['TARGET'] == 0
    condition2 = data['TARGET'] == 1

    # Apply the condition to segment the DataFrame
    segmented_df1 = data[condition1]
    segmented_df2 = data[condition2]

    # Set index to False (reset index)
    segmented_df1 = segmented_df1.reset_index(drop=True)
    segmented_df2 = segmented_df2.reset_index(drop=True)

    # Create histograms
    plt.figure(figsize=(8, 8))

    # Plot the first histogram
    plt.subplot(2, 1, 1)
    plt.hist(segmented_df1["AMT_CREDIT"], bins=30, color='limegreen', alpha=0.7, label='Repaid')
    plt.ticklabel_format(axis='x', style='plain')
    plt.title('Repaid')

    # Plot the second histogram
    plt.subplot(2, 1, 2)
    plt.hist(segmented_df2["AMT_CREDIT"], bins=30, color='orangered', alpha=0.7, label='Defaulted', bottom=5)
    plt.ticklabel_format(axis='x', style='plain')
    plt.title('Defaulted')

    # Display the plots using Streamlit
    st.pyplot(plt)

def create_histogram_income():

    data = df_clients_target[["TARGET", "AMT_INCOME_TOTAL"]]

    # Condition to segment the DataFrame 
    condition1 = data['TARGET'] == 0
    condition2 = data['TARGET'] == 1

    # Apply the condition to segment the DataFrame
    segmented_df1 = data[condition1]
    segmented_df2 = data[condition2]

    # Set index to False (reset index)
    segmented_df1 = segmented_df1.reset_index(drop=True)
    segmented_df2 = segmented_df2.reset_index(drop=True)

    # Create histograms
    plt.figure(figsize=(8, 8))

    x_limit = (0, 1000000)

    # Plot the first histogram
    plt.subplot(2, 1, 1)
    plt.hist(segmented_df1["AMT_INCOME_TOTAL"], bins=30, color='limegreen', alpha=0.7, label='Repaid', range=x_limit)
    plt.ticklabel_format(axis='x', style='plain')
    plt.title('Repaid')

    # Plot the second histogram
    plt.subplot(2, 1, 2)
    plt.hist(segmented_df2["AMT_INCOME_TOTAL"], bins=30, color='orangered', alpha=0.7, label='Defaulted', range=x_limit, bottom=5)
    plt.ticklabel_format(axis='x', style='plain')
    plt.title('Defaulted')

    # Display the plots using Streamlit
    st.pyplot(plt)


@st.cache_data 
def plot_repay(data: dict):

    fig, ax = plt.subplots(figsize=(4, 4))

    colors = ['limegreen', 'orangered']

    ax.pie(data.values(), labels=['Repaied', 'Defaulted'], autopct='%1.1f%%', radius=1, startangle=90, textprops={'fontsize': 8}, colors=colors)
    inner_circle = plt.Circle((0, 0), 0.4, color='white')
    ax.add_artist(inner_circle)

    ax.axis('equal')

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

@st.cache_data 
def get_working():
    response = requests.get(API_ADDRESS + '/api/clients/working')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data 
def plot_working(data: dict):

    image1 = Image.open(path+"/working.png")
    image2 = Image.open(path+"/unemployment.png")

    # Create a Matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))

    # Plot the first image on the left subplot
    ax1.imshow(np.array(image1))
    ax1.text(0.5, -0.1, f"Has a job: {list(data.values())[0]:.2f}%", size=8, ha="center", transform=ax1.transAxes)

    # Plot the second image on the right subplot
    ax2.imshow(np.array(image2))
    ax2.text(0.5, -0.1, f"Does not have a job: {list(data.values())[1]:.2f}%", size=8, ha="center", transform=ax2.transAxes)

    # Remove axis labels
    ax1.axis('off')
    ax2.axis('off')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

@st.cache_data 
def get_married():
    response = requests.get(API_ADDRESS + '/api/clients/married')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data 
def plot_married(data: dict):

    image1 = Image.open(path+"/rings.png")
    image2 = Image.open(path+"/single.png")

    # Create a Matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))

    # Plot the first image on the left subplot
    ax1.imshow(np.array(image1))
    ax1.text(0.5, -0.1, f"Is married: {list(data.values())[0]:.2f}%", size=8, ha="center", transform=ax1.transAxes)

    # Plot the second image on the right subplot
    ax2.imshow(np.array(image2))
    ax2.text(0.5, -0.1, f"Is not married: {list(data.values())[1]:.2f}%", size=8, ha="center", transform=ax2.transAxes)

    # Remove axis labels
    ax1.axis('off')
    ax2.axis('off')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

@st.cache_data 
def get_children():
    response = requests.get(API_ADDRESS + '/api/clients/children')
    if response.status_code == 200: 
        data = response.json()
        return data 
    else:
        st.error('Failed to get clients')
        return None

@st.cache_data 
def plot_children(data: dict):

    image1 = Image.open(path+"/father-and-son.png")
    image2 = Image.open(path+"/parents.png")

    # Create a Matplotlib figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))

    # Plot the first image on the left subplot
    ax1.imshow(np.array(image1))
    ax1.text(0.5, -0.1, f"Has children: {list(data.values())[0]:.2f}%", size=8, ha="center", transform=ax1.transAxes)

    # Plot the second image on the right subplot
    ax2.imshow(np.array(image2))
    ax2.text(0.5, -0.1, f"Does not have children: {list(data.values())[1]:.2f}%", size=8, ha="center", transform=ax2.transAxes)

    # Remove axis labels
    ax1.axis('off')
    ax2.axis('off')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)

@st.cache_data 
def plot_gender_stats():
    response = requests.get(API_ADDRESS + '/api/statistics/genders')

    data = response.json()

    index_values = ['0', '1', '2', '3']

    data = pd.DataFrame(data,  index=index_values)

    # Pivot the DataFrame
    pivot_df = data.pivot(index='Gender', columns='Stats Loan', values='Value')

    # Set specific colors for each category
    category_colors = {'Repaid': 'limegreen', 'Defaulted': 'orangered'}

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot_df.plot(kind='bar', ax=ax, color=[category_colors[col] for col in pivot_df.columns], rot=0)

    st.pyplot(fig)

@st.cache_data 
def plot_house_stats():
    response = requests.get(API_ADDRESS + '/api/statistics/houses')
    
    data = response.json()

    index_values = ['0', '1', '2', '3']

    data = pd.DataFrame(data,  index=index_values)

    # Pivot the DataFrame
    pivot_df = data.pivot(index='Real State Property', columns='Stats Loan', values='Value')

    # Set specific colors for each category
    category_colors = {'Repaid': 'limegreen', 'Defaulted': 'orangered'}

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot_df.plot(kind='bar', ax=ax, color=[category_colors[col] for col in pivot_df.columns], rot=0)
    

    st.pyplot(fig)


@st.cache_data 
def plot_car_stats():
    response = requests.get(API_ADDRESS + '/api/statistics/cars')

    data = response.json()

    index_values = ['0', '1', '2', '3']

    data = pd.DataFrame(data,  index=index_values)

    # Pivot the DataFrame
    pivot_df = data.pivot(index='Vehicle', columns='Stats Loan', values='Value')

    # Set specific colors for each category
    category_colors = {'Repaid': 'limegreen', 'Defaulted': 'orangered'}

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot_df.plot(kind='bar', ax=ax, color=[category_colors[col] for col in pivot_df.columns], rot=0)

    st.pyplot(fig)

@st.cache_data 
def plot_working_stats():
    response = requests.get(API_ADDRESS + '/api/statistics/working')

    data = response.json()

    index_values = ['0', '1', '2', '3']

    data = pd.DataFrame(data,  index=index_values)

    # Pivot the DataFrame
    pivot_df = data.pivot(index='Working status', columns='Stats Loan', values='Value')

    # Set specific colors for each category
    category_colors = {'Repaid': 'limegreen', 'Defaulted': 'orangered'}

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot_df.plot(kind='bar', ax=ax, color=[category_colors[col] for col in pivot_df.columns], rot=0)

    st.pyplot(fig)

@st.cache_data 
def plot_married_stats():
    response = requests.get(API_ADDRESS + '/api/statistics/married')

    data = response.json()

    index_values = ['0', '1', '2', '3']

    data = pd.DataFrame(data,  index=index_values)

    # Pivot the DataFrame
    pivot_df = data.pivot(index='Marital status', columns='Stats Loan', values='Value')

    # Set specific colors for each category
    category_colors = {'Repaid': 'limegreen', 'Defaulted': 'orangered'}

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot_df.plot(kind='bar', ax=ax, color=[category_colors[col] for col in pivot_df.columns], rot=0)

    st.pyplot(fig)

@st.cache_data 
def plot_children_stats():
    response = requests.get(API_ADDRESS + '/api/statistics/children')

    data = response.json()

    index_values = ['0', '1', '2', '3']

    data = pd.DataFrame(data,  index=index_values)

    # Pivot the DataFrame
    pivot_df = data.pivot(index='Children', columns='Stats Loan', values='Value')

    # Set specific colors for each category
    category_colors = {'Repaid': 'limegreen', 'Defaulted': 'orangered'}

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot_df.plot(kind='bar', ax=ax, color=[category_colors[col] for col in pivot_df.columns], rot=0)

    st.pyplot(fig)

tab1, tab2, tab3 = st.tabs(['📊 About the project', 
                            '👥 General customer information', 
                            '❓ Help'
                        ])

with tab1:

    mygrid = make_grid(2,1)
    
    with mygrid[0][0]:

        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(logo, width=200)  

    with mygrid[1][0]:

        st.subheader("Implement a scoring model - - OpenClassrooms Data Scientist P7")
        st.markdown("This project is part of [OpenClassRooms Data Scientist training](https://openclassrooms.com/fr/paths/164-data-scientist)"\
                    " and has the following objectives:")

        objectives_list = '<ul style="list-style-type:disc;">'\
                            '<li>Build a scoring model that will automatically predicts the probability of a client paying a loan.<br>'\
                            'The mision will be treated as a <strong>binary classification problem</strong>.<br>So, 0 will be the class who repaid/pay '\
                            'the loan and 1 will be the class who did not repay/pay the loan.</li>'\
                            '<li>Build an <strong>interactive dashboard</strong> for customer relationship managers to interpret the predictions made by the model,' \
                            'and to improve the customer knowledge of customer relationship managers.</li>'\
                            '<li>Put the <strong>prediction scoring model into production using an API, as well as the interactive dashboard</strong> that calls the API for predictions.</li>'\
                        '</ul>'
        st.markdown(objectives_list, unsafe_allow_html=True)
        
        st.subheader("How to use it ?")
        st.markdown("This dashboard has three pages:")

        how_to_use_text = '<ul style="list-style-type:disc;">'\
                            '<li>The first page contains information about the project, general information about the clients in the database and a help tab.</li>'\
                            '<li>The second page allows you to perform a local analysis. In other words, it provides information on a specific client selected for analysis. The information provided includes: Information about the customer, information about whether the credit should be authorized to that customer and what the determining variables were for authorizing or rejecting the credit.</li>'\
                            '<li>Finally, the third page provides a global analysis. In other words, information relating to a group of similar clients, as well as a list of the most important variables for making a decision: whether a client will have their credit approved or not.</li>'\
                        '</ul>'
        st.markdown(how_to_use_text, unsafe_allow_html=True)

        st.subheader("Other information")
        other_text = '<ul style="list-style-type:disc;">'\
                        '<li><h4>Data</h4>'\
                        'The data used to develop this project are based on the Kaggle\'s</a> competition: '\
                        '<a href="https://www.kaggle.com/c/home-credit-default-risk/overview" target="_blank">Home Credit - Default Risk</a></li>'\
                        '<li><h4>Repository</h4>'\
                        'You can find more information about the project\'s code in its <a href="https://github.com/laisar/OCs_DataScientist_P7" target="_blank">Github\' repository</a></li>'\
                    '</ul>'
        st.markdown(other_text, unsafe_allow_html=True)

with tab2:

    st.subheader("General statistics")
    st.markdown("Below you can see general statistics on our current customers.") 

    st.markdown(""" <h4 style="text-align: center;">General distributions</h4> """, unsafe_allow_html=True)
    st.markdown('\n')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**Loan repayment distribution:**')
        repay  = get_repay()
        plot_repay(repay)
        st.markdown('\n')

    with col2:
        st.markdown('**Total credit repayment distribution:**')
        create_histogram_credit()
        st.markdown('\n')

    with col3:
        st.markdown('**Total income repayment distribution:**')
        create_histogram_income()
        st.markdown('\n')

    st.markdown(""" <h4 style="text-align: center;">Percentages</h4> """, unsafe_allow_html=True)
    st.markdown('\n')

    col4, col5, col6 = st.columns(3)

    with col4:
        gender  = get_gender()
        plot_gender(gender)
        st.markdown('\n')

    with col5:
        house  = get_house()
        plot_house(house)
        st.markdown('\n')

    with col6:
        car  = get_car()
        plot_car(car)
        st.markdown('\n')

    st.markdown(""" <h4 style="text-align: center;">Distributions relative to percentages</h4> """, unsafe_allow_html=True)
    st.markdown('\n')    

    col7, col8, col9 = st.columns(3)

    with col7:
        st.markdown('**Gender repayment distribution:**')
        plot_gender_stats()
        st.markdown('\n')

    with col8:
        st.markdown('**Owns a real state property repayment distribution:**')
        plot_house_stats()
        st.markdown('\n')

    with col9:
        st.markdown('**Owns a vehicle repayment distribution:**')
        plot_car_stats()
        st.markdown('\n')

    st.markdown(""" <h4 style="text-align: center;">Percentages</h4> """, unsafe_allow_html=True)
    st.markdown('\n') 

    col10, col11, col12 = st.columns(3)

    with col10:
        workclient  = get_working()
        plot_working(workclient)
        st.markdown('\n')

    with col11:
        married  = get_married()
        plot_married(married)
        st.markdown('\n')

    with col12:
        children  = get_children()
        plot_children(children)
        st.markdown('\n')

    st.markdown(""" <h4 style="text-align: center;">Distributions relative to percentages</h4> """, unsafe_allow_html=True)
    st.markdown('\n')    

    col13, col14, col15 = st.columns(3)

    with col13:
        st.markdown('**Has job repayment distribution:**')
        plot_working_stats()
        st.markdown('\n')

    with col14:
        st.markdown('**Marital status repayment distribution:**')
        plot_married_stats()
        st.markdown('\n')

    with col15:
        st.markdown('**Has children repayment distribution:**')
        plot_children_stats()
        st.markdown('\n')

with tab3:

    mygrid = make_grid(2,1)
    
    with mygrid[0][0]:

        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(logo, width=200)  

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

