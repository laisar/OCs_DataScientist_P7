# python base image in the container from Docker Hub
FROM python:3.11.5-slim

# copy files to the /app folder in the container
COPY ./fastapi_predict.py /app/fastapi_predict.py
COPY ./Pipfile /app/Pipfile
COPY ./Pipfile.lock /app/Pipfile.lock
COPY ./dataset_predict_compressed.gz /app/dataset_predict_compressed.gz

# set the working directory in the container to be /app
WORKDIR /app

# install the packages from the Pipfile in the container
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile

# expose the port that uvicorn will run the app on
ENV PORT=8000
EXPOSE 8000

# execute the command python fastapi_predict.py (in the WORKDIR) to start the app
CMD ["python", "fastapi_predict.py"]