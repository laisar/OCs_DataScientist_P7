import io
import os
import json 
import gc
import pandas as pd
import numpy as np
from datetime import date, timedelta
from fastapi import FastAPI, File, HTTPException
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import joblib


app = FastAPI(
    title="Home Credit Default Risk",
    description="""Obtain information related to probability of a client defaulting on loan."""
)


def calculate_years(days):
    """
    Method used to calculate years based on date (today - quantity of days).

    Parameters:
    -----------------
        days (int): Numbers of day to rest of today

    Returns:
    -----------------
        years (int): Numbers of years
    """

    today = date.today()
    initial_date = today - timedelta(abs(days))
    years = today.year - initial_date.year - ((today.month, today.day) < (initial_date.month, initial_date.day))

    return years


########################################################
# Columns to read on CSVs
########################################################
COLUMNS = [
    "SK_ID_CURR", "AMT_INCOME_TOTAL", "CODE_GENDER", 
    "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_EMPLOYED", 
    "AMT_CREDIT", "AMT_GOODS_PRICE", "EXT_SOURCE_2",
    "EXT_SOURCE_3", 
]


########################################################
# Reading the csv
########################################################
df_clients_to_predict = pd.read_csv("dataset_predict.csv")
df_current_clients = pd.read_csv("dataset_target.csv")

df_current_clients["AGE"] = df_current_clients["DAYS_BIRTH"].apply(lambda x: calculate_years(x))
df_current_clients["YEARS_EMPLOYED"] = df_current_clients["DAYS_EMPLOYED"].apply(lambda x: calculate_years(x))
df_current_clients["EXT_SOURCE_2"] = df_current_clients["EXT_SOURCE_2"].round(3)
df_current_clients["EXT_SOURCE_3"] = df_current_clients["EXT_SOURCE_3"].round(3)

df_current_clients_by_target_repaid = df_current_clients[df_current_clients["TARGET"] == 0]
df_current_clients_by_target_not_repaid = df_current_clients[df_current_clients["TARGET"] == 1]



@app.get("/api/clients")
async def clients_id():
    """ 
    EndPoint to get all clients id
    """
    
    clients_id = df_clients_to_predict["SK_ID_CURR"].tolist()

    return {"clientsId": clients_id}


@app.get("/api/predictions/clients/{id}")
async def predict(id: int):
    """ 
    EndPoint to get the probability honor/compliance of a client
    """ 

    clients_id = df_clients_to_predict["SK_ID_CURR"].tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail="client's id not found")
    else:
        # Loading the model
        model = joblib.load("models/lightgbm_model.pkl")

        threshold = 0.174

        # Filtering by client's id
        df_prediction_by_id = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]
        df_prediction_by_id = df_prediction_by_id.drop(df_prediction_by_id.columns[[0, 1]], axis=1)

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


@app.get("/api/predictions/clients/shap/{id}")
async def client_shap_df(id: int):
    """ 
    EndPoint to return a df with all client's data
    """ 
    
    clients_id = df_clients_to_predict["SK_ID_CURR"].tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail="client's id not found")
    else:

        # Filtering by client's id
        idx = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"]==id].index[0]

        client = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id].drop(columns=["SK_ID_CURR", "AMT_INCOME_TOTAL"])
        client = client.to_json(orient="records")

    return client


@app.get("/api/statistics/ages")
async def statistical_age():
    """ 
    EndPoint to get some statistics - ages
    """

    ages_data_repaid = df_current_clients_by_target_repaid.groupby("AGE").size()
    ages_data_repaid = pd.DataFrame(ages_data_repaid).reset_index()
    ages_data_repaid.columns = ["AGE", "AMOUNT"]
    ages_data_repaid = ages_data_repaid.set_index("AGE").to_dict()["AMOUNT"]

    ages_data_not_repaid = df_current_clients_by_target_not_repaid.groupby("AGE").size()
    ages_data_not_repaid = pd.DataFrame(ages_data_not_repaid).reset_index()
    ages_data_not_repaid.columns = ["AGE", "AMOUNT"]
    ages_data_not_repaid = ages_data_not_repaid.set_index("AGE").to_dict()["AMOUNT"]

    return {"ages_repaid" : ages_data_repaid, "ages_not_repaid" : ages_data_not_repaid}


@app.get("/api/statistics/yearsEmployed")
async def statistical_years_employed():
    """ 
    EndPoint to get some statistics - years employed
    """

    years_employed_data_repaid = df_current_clients_by_target_repaid.groupby("YEARS_EMPLOYED").size()
    years_employed_data_repaid = pd.DataFrame(years_employed_data_repaid).reset_index()
    years_employed_data_repaid.columns = ["YEARS_EMPLOYED", "AMOUNT"]
    years_employed_data_repaid = years_employed_data_repaid.set_index("YEARS_EMPLOYED").to_dict()["AMOUNT"]

    years_employed_data_not_repaid = df_current_clients_by_target_not_repaid.groupby("YEARS_EMPLOYED").size()
    years_employed_data_not_repaid = pd.DataFrame(years_employed_data_not_repaid).reset_index()
    years_employed_data_not_repaid.columns = ["YEARS_EMPLOYED", "AMOUNT"]
    years_employed_data_not_repaid = years_employed_data_not_repaid.set_index("YEARS_EMPLOYED").to_dict()["AMOUNT"]

    return {
        "years_employed_repaid" : years_employed_data_repaid,
        "years_employed_not_repaid" : years_employed_data_not_repaid
    }


@app.get("/api/statistics/amtCredits")
async def statistical_amt_credit():
    """ 
    EndPoint to get some statistics - AMT Credit
    """

    amt_credit_data_repaid = df_current_clients_by_target_repaid.groupby("AMT_CREDIT").size()
    amt_credit_data_repaid = pd.DataFrame(amt_credit_data_repaid).reset_index()
    amt_credit_data_repaid.columns = ["AMT_CREDIT", "AMOUNT"]
    amt_credit_data_repaid = amt_credit_data_repaid.set_index("AMT_CREDIT").to_dict()["AMOUNT"]

    amt_credit_data_not_repaid = df_current_clients_by_target_not_repaid.groupby("AMT_CREDIT").size()
    amt_credit_data_not_repaid = pd.DataFrame(amt_credit_data_not_repaid).reset_index()
    amt_credit_data_not_repaid.columns = ["AMT_CREDIT", "AMOUNT"]
    amt_credit_data_not_repaid = amt_credit_data_not_repaid.set_index("AMT_CREDIT").to_dict()["AMOUNT"]

    return {
        "amt_credit_repaid" : amt_credit_data_repaid,
        "amt_credit_not_repaid" : amt_credit_data_not_repaid
    }


@app.get("/api/statistics/amtIncomes")
async def statistical_amt_income():
    """ 
    EndPoint to get some statistics - AMT Income
    """

    amt_income_data_repaid = df_current_clients_by_target_repaid.groupby("AMT_INCOME_TOTAL").size()
    amt_income_data_repaid = pd.DataFrame(amt_income_data_repaid).reset_index()
    amt_income_data_repaid.columns = ["AMT_INCOME", "AMOUNT"]
    amt_income_data_repaid = amt_income_data_repaid.set_index("AMT_INCOME").to_dict()["AMOUNT"]

    amt_income_data_not_repaid = df_current_clients_by_target_not_repaid.groupby("AMT_INCOME_TOTAL").size()
    amt_income_data_not_repaid = pd.DataFrame(amt_income_data_not_repaid).reset_index()
    amt_income_data_not_repaid.columns = ["AMT_INCOME", "AMOUNT"]
    amt_income_data_not_repaid = amt_income_data_not_repaid.set_index("AMT_INCOME").to_dict()["AMOUNT"]

    return {
        "amt_income_repaid" : amt_income_data_repaid,
        "amt_income_not_repaid" : amt_income_data_not_repaid
    }

@app.get("/api/statistics/extSource2")
async def statistical_ext_source_2():
    """ 
    EndPoint to get some statistics - EXT SOURCE 2
    """

    ext_source_2_data_repaid = df_current_clients_by_target_repaid.groupby("EXT_SOURCE_2").size()
    ext_source_2_data_repaid = pd.DataFrame(ext_source_2_data_repaid).reset_index()
    ext_source_2_data_repaid.columns = ["EXT_SOURCE_2", "AMOUNT"]
    ext_source_2_data_repaid = ext_source_2_data_repaid.set_index("EXT_SOURCE_2").to_dict()["AMOUNT"]

    ext_source_2_data_not_repaid = df_current_clients_by_target_not_repaid.groupby("EXT_SOURCE_2").size()
    ext_source_2_data_not_repaid = pd.DataFrame(ext_source_2_data_not_repaid).reset_index()
    ext_source_2_data_not_repaid.columns = ["EXT_SOURCE_2", "AMOUNT"]
    ext_source_2_data_not_repaid = ext_source_2_data_not_repaid.set_index("EXT_SOURCE_2").to_dict()["AMOUNT"]

    return {
        "ext_source_2_repaid" : ext_source_2_data_repaid,
        "ext_source_2_not_repaid" : ext_source_2_data_not_repaid
    }


@app.get("/api/statistics/extSource3")
async def statistical_ext_source_3():
    """ 
    EndPoint to get some statistics - EXT SOURCE 3
    """

    ext_source_3_data_repaid = df_current_clients_by_target_repaid.groupby("EXT_SOURCE_3").size()
    ext_source_3_data_repaid = pd.DataFrame(ext_source_3_data_repaid).reset_index()
    ext_source_3_data_repaid.columns = ["EXT_SOURCE_3", "AMOUNT"]
    ext_source_3_data_repaid = ext_source_3_data_repaid.set_index("EXT_SOURCE_3").to_dict()["AMOUNT"]

    ext_source_3_data_not_repaid = df_current_clients_by_target_not_repaid.groupby("EXT_SOURCE_3").size()
    ext_source_3_data_not_repaid = pd.DataFrame(ext_source_3_data_not_repaid).reset_index()
    ext_source_3_data_not_repaid.columns = ["EXT_SOURCE_3", "AMOUNT"]
    ext_source_3_data_not_repaid = ext_source_3_data_not_repaid.set_index("EXT_SOURCE_3").to_dict()["AMOUNT"]

    return {
        "ext_source_3_repaid" : ext_source_3_data_repaid,
        "ext_source_3_not_repaid" : ext_source_3_data_not_repaid
    }