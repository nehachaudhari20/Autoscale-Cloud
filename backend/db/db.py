import psycopg2
import os

def get_connection():
    conn = psycopg2.connect(
        host="autoscale-db.cwngcomm4d6a.us-east-1.rds.amazonaws.com",
        database="postgres",
        user="postgres",
        password="testrdsaws",
        port=5432,
        sslmode="verify-ca",
        sslrootcert=os.path.join(os.path.dirname(__file__), "rds-ca.pem")
    )
    return conn