# OCs_DataScientist_P7
Project carried out as part of the OpenClassrooms Data Scientist training - Project 7: __Implémentez un modèle de scoring__

## **Context**

"Prêt à dépenser" offers consumer credit for people with little or no loan history.  The company wishes to implement a “credit scoring” tool to calculate the probability that a customer will repay their credit, and then classify the request into credit granted or refused.

The company therefore wishes to develop a classification algorithm based on various data sources. In addition, customer relations managers have highlighted the fact that customers are increasingly demanding transparency regarding credit granting decisions. 

"Prêt à dépenser" decides to develop an interactive dashboard so that customer relationship managers can both explain credit granting decisions as transparently as possible and also allow their customers to have access to their personal information and explore them easily.

## **Mission **

1. Build a scoring model that will automatically predict a customer's probability of bankruptcy.
2. Build an interactive dashboard for customer relationship managers to interpret the predictions made by the model, and to improve the customer knowledge of customer relationship managers.
3. Put the prediction scoring model into production using an API, as well as the interactive dashboard that calls the API for predictions.

* For detailed information in French about the project, required deliverables, and evaluation criteria [click here](https://drive.google.com/file/d/1kiqlS2SoUQB9ncG0wrmKFuKfOJxZQVk7/view?usp=sharing).
* The original data used in the project can be downloaded from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)

## **Project organization**

------------

    ├── README.md
    │
    ├── datadrift                <- Datadrift report
    │
    ├── fastAPI                  <- FastAPI related files
    │
    ├── models                   <- LightGBM and SHAP explainer
    │
    ├── streamlit                <- Dashboard related files
    │
    │   ├── images               <- Images used in dashboard
    │   ├── pages                <- Dashboard pages
    |
    ├── dataset_predict.csv      <- Dataset with Target NaNs  
    │
    ├── dataset_target.csv       <- Dataset used to modelling
    │
    ├── modelling.ipynb          <- Modelling python code
    │
    ├── preprocessing.ipynb      <- Preprocessing python code

## **Dashboard**

The dashboard contains the following features:

- It allows you to visualize the score and the interpretation of this score for each client in an intelligible way for a person who is not an expert in data science.
- It allows you to view descriptive information relating to a customer.
- It allows you to compare descriptive information relating to a customer to all customers or to a group of similar customers.

Click here to access the dashboard




