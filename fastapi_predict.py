import io
import os
import json
import uvicorn 
import gc
import pandas as pd
import numpy as np
from datetime import date, timedelta
from fastapi import FastAPI, File, HTTPException
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
from sklearn.neighbors import NearestNeighbors
import joblib
import pickle


app = FastAPI(
    title="Home Credit Default Risk",
    description="""Obtain information related to probability of a client defaulting on loan."""
)

########################################################
# Reading the csv
########################################################
df_clients_to_predict = pd.read_csv("dataset_predict.csv")
#df_clients_to_predict_original = pd.read_csv("dataset_predict_original.csv")
df_current_clients = pd.read_csv("dataset_target.csv")


@app.get("/api/test")
async def test():
    """ 
    EndPoint to get all clients id
    """
    
    test = "Hello"

    return test

@app.get("/api/clients")
async def clients_id():
    """ 
    EndPoint to get all clients id
    """
    
    clients_id = df_clients_to_predict["SK_ID_CURR"].tolist()

    return {"clientsId": clients_id}


@app.get("/api/predictions/clients")
async def predict(id: int):
    """ 
    EndPoint to get the probability honor/compliance of a client
    """ 

    clients_id = df_clients_to_predict["SK_ID_CURR"].tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail="client's id not found")
    else:
        # Loading the model
        model = joblib.load("models/lightgbm_model.pckl")

        threshold = 0.365

        # Filtering by client's id
        df_prediction_by_id = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]
        df_prediction_by_id = df_prediction_by_id.drop(columns=["SK_ID_CURR", "TARGET", "REPAY", "CLUSTER"])

        # Predicting
        result_proba = model.predict_proba(df_prediction_by_id)
        y_prob = result_proba[:, 1]
        
        result = (y_prob >= threshold).astype(int)

        if (int(result[0]) == 0):
            result = "Yes"
        else:
            result = "No"    

    return {
        "repay" : result,
        "probability0" : result_proba[0][0],
        "probability1" : result_proba[0][1],
        "threshold" : threshold
    }

@app.get("/api/predictions/list_clients")
async def clients_list(id: bool):
    """ 
    EndPoint to get clients that repay and clients that do not repay
    """ 
    if id:  
        result = df_clients_to_predict.loc[df_clients_to_predict['REPAY'] == True, 'SK_ID_CURR'].tolist()
    else:
        result = df_clients_to_predict.loc[df_clients_to_predict['REPAY'] == False, 'SK_ID_CURR'].tolist() 

    return result

@app.get("/api/clients/clients_info")
async def client_info(id: int):

    """ 
    EndPoint to get client's detail 
    """
     
    # Filtering by client's id
    client = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]

    client_information = {
        
        "Client ID": id,
        "Total credit amount": round(client['AMT_CREDIT'].values[0]),
        "Credit amount repaied per year": round(client['AMT_ANNUITY'].values[0]),
        "Client anual income": round(client['AMT_INCOME_TOTAL'].values[0]),
        "Payment rate (%)": round(client['PAYMENT_RATE'].values[0]*100),
        "Source 2 (%)": round(client['EXT_SOURCE_2'].values[0]*100),
        "Source 3 (%)": round(client['EXT_SOURCE_3'].values[0]*100),
        "Gender": "Man" if int(client['CODE_GENDER'].values[0]) == 0 else "Woman",
        "Age": round(abs(client['DAYS_BIRTH'].values[0]/365)),
        "Married": "No" if int(client['NAME_FAMILY_STATUS_Married'].values[0]) == 0 else "Yes",
        "Number of children": int(client['CNT_CHILDREN'].values[0]),
        "Working": "No" if int(client['NAME_INCOME_TYPE_Working'].values[0]) == 0 else "Yes",
        "Working since": str(round(abs(client['DAYS_EMPLOYED'].values[0]/365)))+" years",
        "Owns a car": "No" if int(client['FLAG_OWN_CAR'].values[0]) == 0 else "Yes",
        "Owns a real estate property": "No" if int(client['FLAG_OWN_REALTY'].values[0]) == 0 else "Yes"
        
    }
    
    return client_information

@app.get("/api/clients/client")
async def explain(id: int):
    
    client = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]
    client = client.drop(columns=["SK_ID_CURR", "TARGET", "REPAY", "CLUSTER"])
    client = client.to_dict(orient="records")
    
    return client

@app.get("/api/clients/similar_clients")
async def similar_clients(id: int):
    
    client = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]
    client = client.drop(columns=["SK_ID_CURR", "TARGET", "REPAY", "CLUSTER"])
    cluster = int(df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]["CLUSTER"])
    df_client_cluster = df_clients_to_predict[df_clients_to_predict["CLUSTER"] == cluster].reset_index(drop=True)
    
    # Choose the number of neighbors (k)
    k = 10

    # Create and fit the KNN model
    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(df_client_cluster.drop(columns=["SK_ID_CURR", "TARGET", "REPAY", "CLUSTER"]))
    distances, indices = knn_model.kneighbors(client)
    
    # Create and fit the KNN model
    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(df_client_cluster.drop(columns=["SK_ID_CURR", "TARGET", "REPAY", "CLUSTER"]))
    distances, indices = knn_model.kneighbors(client)
    
    df_similar_clients = df_client_cluster.iloc[indices[0]].reset_index(drop=True)

    similar_clients = df_similar_clients["SK_ID_CURR"].tolist()
    
    return similar_clients

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000) 