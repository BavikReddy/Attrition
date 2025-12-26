ARG ECR_ENDPOINT
FROM $ECR_ENDPOINT/sagemaker-xgboost:0.90-1-cpu-py3


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY queries /opt/program/queries
COPY featureset /opt/program
COPY db_connect.py /opt/program
COPY requirements.txt /opt/program
WORKDIR /opt/program

RUN pip install -r requirements.txt
