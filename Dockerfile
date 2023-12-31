# python base image in the container from Docker Hub
FROM python:3.11.5-slim

# copy files to the /app folder in the container
COPY ./fastapi_predict.py /app/fastapi_predict.py
COPY ./data/dataset_predict_compressed.gz /app/data/dataset_predict_compressed.gz
COPY ./data/dataset_target_compressed.gz /app/data/dataset_target_compressed.gz
COPY ./models/xgboost_classifier.pkl /app/models/xgboost_classifier.pkl
COPY ./models/xgboost_shap_explainer.pkl /app/models/xgboost_shap_explainer.pkl
COPY ./Pipfile /app/Pipfile
COPY ./Pipfile.lock /app/Pipfile.lock

# set the working directory in the container to be /app
WORKDIR /app

# install the packages from the Pipfile in the container
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile

# expose the port that uvicorn will run the app on
ENV PORT=8000
EXPOSE 8000

# execute the command python main.py (in the WORKDIR) to start the app
CMD ["python", "fastapi_predict.py"]