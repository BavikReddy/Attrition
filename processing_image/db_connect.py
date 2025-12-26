import os
import json
import boto3
from botocore.config import Config
from sqlalchemy.engine import create_engine


def setup_db_engine():

    cfg = Config(region_name='us-east-1')
    client = boto3.client('secretsmanager',config=cfg)
    secret = json.loads(client.get_secret_value(SecretId='edl-analytics/redshift_data_api')['SecretString'])

    db_config = {
        'usr': secret['username'],
        'pass': secret['password'],
        'host': secret['host'],
        'port': secret['port'],
        'db': 'edl' # secret['db']
    }
    return create_engine('postgresql://{usr}:{pass}@{host}:{port}/{db}'.format(**db_config))


def read_sql_query(path_to_query):

    with open(os.path.join(os.path.dirname(__file__), path_to_query), 'r') as f:
        sql_new_data = f.read()
    return sql_new_data