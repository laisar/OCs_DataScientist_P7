import uvicorn 
import pandas as pd
from datetime import date, timedelta
from fastapi import FastAPI, File, HTTPException
from fastapi.responses import JSONResponse
from sklearn.neighbors import NearestNeighbors
import joblib
import pickle
import os
import numpy as np

app = FastAPI(
    title=" Home Credit Default Risk ",
    description="""Obtain information related to probability of a client defaulting on loan."""
)

# load environment variables
#port = os.environ["PORT"]
########################################################
# Reading the csvp
########################################################
df_clients_to_predict = pd.read_csv("./data/dataset_predict_compressed.gz", compression='gzip', sep=',')

model = pickle.load(open("./models/lightgbm_model.pckl", 'rb'))

# Load shap model
lgbm_shap = pickle.load(open("./models/shap_explainer.pckl", 'rb'))
shap_values = lgbm_shap.shap_values(df_clients_to_predict.drop(columns=["SK_ID_CURR", "TARGET", "REPAY"]))


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

        threshold = 0.426

        # Filtering by client's id
        df_prediction_by_id = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]
        df_prediction_by_id = df_prediction_by_id.drop(columns=["SK_ID_CURR", "TARGET", "REPAY"])

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
    client = client.drop(columns=["SK_ID_CURR", "TARGET", "REPAY"])
    client = client.to_dict(orient="records")
    
    return client

@app.get("/api/clients/similar_clients")
async def similar_clients(id: int):
    
    df_client_cluster = df_clients_to_predict
    client = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]
    client = client.drop(columns=["SK_ID_CURR", "TARGET", "REPAY"])
    
    # Choose the number of neighbors (k)
    k = 15

    # Create and fit the KNN model
    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(df_client_cluster.drop(columns=["SK_ID_CURR", "TARGET", "REPAY"]))
    distances, indices = knn_model.kneighbors(client)
    
    # Create and fit the KNN model
    knn_model = NearestNeighbors(n_neighbors=k)
    knn_model.fit(df_client_cluster.drop(columns=["SK_ID_CURR", "TARGET", "REPAY"]))
    distances, indices = knn_model.kneighbors(client)
    
    df_similar_clients = df_client_cluster.iloc[indices[0]].reset_index(drop=True)

    similar_clients = df_similar_clients["SK_ID_CURR"].tolist()
    
    return similar_clients

@app.get('/api/clients/prediction/shap/local')
async def get_local_shap(id: int):
    ''' Endpoint to get local shap values
    '''
    clients_id = df_clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    else:
        idx = int(list(df_clients_to_predict[df_clients_to_predict['SK_ID_CURR'] == id].index.values)[0])

        shap_values_idx = shap_values[0][idx, :]
        shap_values_abs_sum = np.abs(shap_values_idx)
        top_feature_indices = np.argsort(shap_values_abs_sum)[-10:]
        top_feature_names = df_clients_to_predict.drop(columns=["SK_ID_CURR", "TARGET", "REPAY"]).columns[top_feature_indices]
        top_feature_shap_values = shap_values_idx[top_feature_indices]


    client_shap = {}

    for name, value in zip(top_feature_names, top_feature_shap_values):
        client_shap[name] = value

    return client_shap

@app.get('/api/clients/prediction/shap/global')
async def get_global_shap(id: int):
    ''' Endpoint to get global shap values
    '''
    clients_id = df_clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    else:
        idx = int(list(df_clients_to_predict[df_clients_to_predict['SK_ID_CURR'] == id].index.values)[0])

        feature_names = df_clients_to_predict.drop(columns=["SK_ID_CURR", "TARGET", "REPAY"]).columns
        shap_values_summary = pd.DataFrame(shap_values[0], columns=feature_names)
        top_global_features = shap_values_summary.abs().mean().nlargest(10)

    client_shap = {}

    client_shap.update(top_global_features.to_dict())

    return client_shap

#test
if __name__ == "__main__":
    uvicorn.run("fastapi_predict:app", reload=True, host="0.0.0.0", port=8000)
