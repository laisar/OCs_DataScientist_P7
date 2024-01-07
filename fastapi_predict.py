import uvicorn 
import pandas as pd
from datetime import date, timedelta
from fastapi import FastAPI, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors
import joblib
import pickle
import json
import os
import numpy as np

app = FastAPI(
    title=" Home Credit Default Risk FastAPI - OpenClassrooms Data Scientist P7",
    description="""Get information about clients, as well as the likelihood of loan default."""
)

# load environment variables
port = os.environ["PORT"]


df_clients_to_predict = pd.read_csv("./data/dataset_predict_compressed.gz", compression='gzip', sep=',')
df_clients_target = pd.read_csv("./data/dataset_target_compressed.gz", compression='gzip', sep=',')

model = pickle.load(open("./models/xgboost_classifier.pkl", 'rb'))

# Load shap model
lgbm_shap = pickle.load(open("./models/xgboost_shap_explainer.pkl", 'rb'))
shap_values = lgbm_shap.shap_values(df_clients_to_predict.drop(columns=["SK_ID_CURR", "TARGET", "REPAY"]))

@app.get("/")
def read_root():
    return {"message": "Home Credit Default Risk FastAPI - OpenClassrooms Data Scientist P7"}


@app.get("/api/clients/gender")
async def clients_gender():
    """ 
    EndPoint to get client's gender
    """

    percentage_m = round((df_clients_target['CODE_GENDER'][df_clients_target['CODE_GENDER'] == 0].count() / len(df_clients_target)) * 100,2)
    percentage_w = round((df_clients_target['CODE_GENDER'][df_clients_target['CODE_GENDER'] == 1].count() / len(df_clients_target)) * 100,2)

    gender_information = {
        
        "percentage_m": percentage_m,
        "percentage_w": percentage_w
    }

    return gender_information

@app.get("/api/clients/repay")
async def clients_repay():
    """ 
    EndPoint to get client's gender
    """

    repaid = round((df_clients_target['TARGET'][df_clients_target['TARGET'] == 0].count() / len(df_clients_target)) * 100,2)
    defaulted = round((df_clients_target['TARGET'][df_clients_target['TARGET'] == 1].count()/ len(df_clients_target)) * 100,2)

    repay_information = {
        
        "Repaid": repaid,
        "Defaulted": defaulted
    }

    return repay_information

@app.get("/api/clients/car")
async def clients_car():
    """ 
    EndPoint to get client's gender
    """

    car = round((df_clients_target['FLAG_OWN_CAR'][df_clients_target['FLAG_OWN_CAR'] == 1].count() / len(df_clients_target)) * 100,2)
    nocar = round((df_clients_target['FLAG_OWN_CAR'][df_clients_target['FLAG_OWN_CAR'] == 0].count()/ len(df_clients_target)) * 100,2)

    car_information = {
        
        "Repaid": car,
        "Defaulted": nocar
    }

    return car_information

@app.get("/api/clients/house")
async def clients_house():
    """ 
    EndPoint to get client's gender
    """

    house = round((df_clients_target['FLAG_OWN_REALTY'][df_clients_target['FLAG_OWN_REALTY'] == 1].count() / len(df_clients_target)) * 100,2)
    nohouse = round((df_clients_target['FLAG_OWN_REALTY'][df_clients_target['FLAG_OWN_REALTY'] == 0].count()/ len(df_clients_target)) * 100,2)

    house_information = {
        
        "Repaid": house,
        "Defaulted": nohouse
    }

    return house_information

@app.get("/api/clients/working")
async def clients_working():
    """ 
    EndPoint to get client's working
    """

    hasajob = round((df_clients_target['NAME_INCOME_TYPE_Working'][df_clients_target['NAME_INCOME_TYPE_Working'] == 1].count() / len(df_clients_target)) * 100,2)
    doesnothasajob = round((df_clients_target['NAME_INCOME_TYPE_Working'][df_clients_target['NAME_INCOME_TYPE_Working'] == 0].count()/ len(df_clients_target)) * 100,2)

    working_info = {
        
        "Repaid": hasajob,
        "Defaulted": doesnothasajob
    }

    return working_info

@app.get("/api/clients/married")
async def clients_married():
    """ 
    EndPoint to get client's married
    """

    married = round((df_clients_target['NAME_FAMILY_STATUS_Married'][df_clients_target['NAME_FAMILY_STATUS_Married'] == 1].count() / len(df_clients_target)) * 100,2)
    notmarried = round((df_clients_target['NAME_FAMILY_STATUS_Married'][df_clients_target['NAME_FAMILY_STATUS_Married'] == 0].count()/ len(df_clients_target)) * 100,2)

    married_info = {
        
        "Repaid": married,
        "Defaulted": notmarried
    }

    return married_info

@app.get("/api/clients/children")
async def clients_children():
    """ 
    EndPoint to get client's children
    """

    haschildren = round((df_clients_target['CNT_CHILDREN'][df_clients_target['CNT_CHILDREN'] >= 1].count() / len(df_clients_target)) * 100,2)
    notchildren = round((df_clients_target['CNT_CHILDREN'][df_clients_target['CNT_CHILDREN'] == 0].count()/ len(df_clients_target)) * 100,2)

    children_info = {
        
        "Repaid": haschildren,
        "Defaulted": notchildren
    }

    return children_info

@app.get('/api/statistics/genders')
async def get_stats_gender():

    count_rows1 = df_clients_target.loc[(df_clients_target['CODE_GENDER'] == 0) & (df_clients_target['TARGET'] == 0)].shape[0]
    count_rows2 = df_clients_target.loc[(df_clients_target['CODE_GENDER'] == 0) & (df_clients_target['TARGET'] == 1)].shape[0]
    count_rows3 = df_clients_target.loc[(df_clients_target['CODE_GENDER'] == 1) & (df_clients_target['TARGET'] == 0)].shape[0]
    count_rows4 = df_clients_target.loc[(df_clients_target['CODE_GENDER'] == 1) & (df_clients_target['TARGET'] == 1)].shape[0]

    data = {'Gender': ['Men', 'Men', 'Women', 'Women'],
            'Stats Loan': ['Repaid', 'Defaulted', 'Repaid', 'Defaulted'],
            'Value': [count_rows1, count_rows2, count_rows3, count_rows4]}

    df = pd.DataFrame(data)

    return JSONResponse(content=df.to_dict(orient='records'), media_type="application/json")

@app.get('/api/statistics/houses')
async def get_stats_house():

    count_rows1 = df_clients_target.loc[(df_clients_target['TARGET'] == 0) & (df_clients_target['FLAG_OWN_REALTY'] == 1)].shape[0]
    count_rows2 = df_clients_target.loc[(df_clients_target['TARGET'] == 1) & (df_clients_target['FLAG_OWN_REALTY'] == 1)].shape[0]
    count_rows3 = df_clients_target.loc[(df_clients_target['TARGET'] == 0) & (df_clients_target['FLAG_OWN_REALTY'] == 0)].shape[0]
    count_rows4 = df_clients_target.loc[(df_clients_target['TARGET'] == 1) & (df_clients_target['FLAG_OWN_REALTY'] == 0)].shape[0]

    data = {'Real State Property': ['Owns property', "Owns property", "Doesn't own property", "Doesn't own property"],
            'Stats Loan': ['Repaid', 'Defaulted', 'Repaid', 'Defaulted'],
            'Value': [count_rows1, count_rows2, count_rows3, count_rows4]}

    df = pd.DataFrame(data)

    return JSONResponse(content=df.to_dict(orient='records'), media_type="application/json")

@app.get('/api/statistics/cars')
async def get_stats_car():

    count_rows1 = df_clients_target.loc[(df_clients_target['TARGET'] == 0) & (df_clients_target['FLAG_OWN_CAR'] == 1)].shape[0]
    count_rows2 = df_clients_target.loc[(df_clients_target['TARGET'] == 1) & (df_clients_target['FLAG_OWN_CAR'] == 1)].shape[0]
    count_rows3 = df_clients_target.loc[(df_clients_target['TARGET'] == 0) & (df_clients_target['FLAG_OWN_CAR'] == 0)].shape[0]
    count_rows4 = df_clients_target.loc[(df_clients_target['TARGET'] == 1) & (df_clients_target['FLAG_OWN_CAR'] == 0)].shape[0]

    data = {'Vehicle': ['Owns vehicle', "Owns vehicle", "Doesn't own vehicle", "Doesn't own vehicle"],
            'Stats Loan': ['Repaid', 'Defaulted', 'Repaid', 'Defaulted'],
            'Value': [count_rows1, count_rows2, count_rows3, count_rows4]}

    df = pd.DataFrame(data)

    return JSONResponse(content=df.to_dict(orient='records'), media_type="application/json")

@app.get('/api/statistics/working')
async def get_stats_work():

    count_rows1 = df_clients_target.loc[(df_clients_target['TARGET'] == 0) & (df_clients_target['NAME_INCOME_TYPE_Working'] == 1)].shape[0]
    count_rows2 = df_clients_target.loc[(df_clients_target['TARGET'] == 1) & (df_clients_target['NAME_INCOME_TYPE_Working'] == 1)].shape[0]
    count_rows3 = df_clients_target.loc[(df_clients_target['TARGET'] == 0) & (df_clients_target['NAME_INCOME_TYPE_Working'] == 0)].shape[0]
    count_rows4 = df_clients_target.loc[(df_clients_target['TARGET'] == 1) & (df_clients_target['NAME_INCOME_TYPE_Working'] == 0)].shape[0]

    data = {'Working status': ['Works', "Works", "Doesn't work", "Doesn't work"],
            'Stats Loan': ['Repaid', 'Defaulted', 'Repaid', 'Defaulted'],
            'Value': [count_rows1, count_rows2, count_rows3, count_rows4]}

    df = pd.DataFrame(data)

    return JSONResponse(content=df.to_dict(orient='records'), media_type="application/json")

@app.get('/api/statistics/married')
async def get_stats_married():

    count_rows1 = df_clients_target.loc[(df_clients_target['TARGET'] == 0) & (df_clients_target['NAME_FAMILY_STATUS_Married'] == 1)].shape[0]
    count_rows2 = df_clients_target.loc[(df_clients_target['TARGET'] == 1) & (df_clients_target['NAME_FAMILY_STATUS_Married'] == 1)].shape[0]
    count_rows3 = df_clients_target.loc[(df_clients_target['TARGET'] == 0) & (df_clients_target['NAME_FAMILY_STATUS_Married'] == 0)].shape[0]
    count_rows4 = df_clients_target.loc[(df_clients_target['TARGET'] == 1) & (df_clients_target['NAME_FAMILY_STATUS_Married'] == 0)].shape[0]

    data = {'Marital status': ['Married', "Married", "Not Married", "Not Married"],
            'Stats Loan': ['Repaid', 'Defaulted', 'Repaid', 'Defaulted'],
            'Value': [count_rows1, count_rows2, count_rows3, count_rows4]}

    df = pd.DataFrame(data)

    return JSONResponse(content=df.to_dict(orient='records'), media_type="application/json")

@app.get('/api/statistics/children')
async def get_stats_children():

    count_rows1 = df_clients_target.loc[(df_clients_target['TARGET'] == 0) & (df_clients_target['CNT_CHILDREN'] >= 1)].shape[0]
    count_rows2 = df_clients_target.loc[(df_clients_target['TARGET'] == 1) & (df_clients_target['CNT_CHILDREN'] >= 1)].shape[0]
    count_rows3 = df_clients_target.loc[(df_clients_target['TARGET'] == 0) & (df_clients_target['CNT_CHILDREN'] == 0)].shape[0]
    count_rows4 = df_clients_target.loc[(df_clients_target['TARGET'] == 1) & (df_clients_target['CNT_CHILDREN'] == 0)].shape[0]

    data = {'Children': ['Has children', "Has children", "Doesn't have children", "Doesn't have children"],
            'Stats Loan': ['Repaid', 'Defaulted', 'Repaid', 'Defaulted'],
            'Value': [count_rows1, count_rows2, count_rows3, count_rows4]}

    df = pd.DataFrame(data)

    return JSONResponse(content=df.to_dict(orient='records'), media_type="application/json")

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
        raise HTTPException(status_code=404, detail="Client id not found")
    else:
        # Loading the model

        threshold = 0.419

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
        "probability0" : result_proba[0][0].tolist(),
        "probability1" : result_proba[0][1].tolist(),
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
    clients_id = df_clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    
    else:
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
    """ 
    EndPoint to get client's complet info
    """

    clients_id = df_clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    
    else:
        client = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]
        client = client.drop(columns=["SK_ID_CURR", "TARGET", "REPAY"])
        client = client.to_dict(orient="records")
    
    return client


@app.get("/api/clients/similar_clients")
async def similar_clients(id: int):
    """ 
    EndPoint to get similar clients
    """

    clients_id = df_clients_to_predict['SK_ID_CURR'].astype(int).tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail='Client id not found')
    
    else:
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

        shap_values_idx = shap_values[idx]
        shap_values_abs_sum = np.abs(shap_values_idx)
        top_feature_indices = np.argsort(shap_values_abs_sum)[-10:]
        top_feature_names = df_clients_to_predict.drop(columns=["SK_ID_CURR", "TARGET", "REPAY"]).columns[top_feature_indices]
        top_feature_shap_values = shap_values_idx[top_feature_indices]


    client_shap = {}

    for i in range(0, 10):
        feature_name = str(top_feature_names[i])  # Ensure the feature name is a string
        shap_value = float(top_feature_shap_values[i])  # Convert Shapley value to float
        client_shap[feature_name] = shap_value

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
        shap_values_summary = pd.DataFrame(shap_values, columns=feature_names)
        top_global_features = shap_values_summary.abs().mean().nlargest(10)

    client_shap = {}

    client_shap.update(top_global_features.to_dict())

    return client_shap

if __name__ == "__main__":
    uvicorn.run("fastapi_predict:app", host="0.0.0.0", port=int(port), reload=False)