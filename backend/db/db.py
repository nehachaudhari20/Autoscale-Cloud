import psycopg2

def get_connection():
    conn = psycopg2.connect(
        host="autoscale-db.cwngcomm4d6a.us-east-1.rds.amazonaws.com",
        database="autoscale-db",
        user="postgres",
        password="testrdsaws",
        port=5432
    )
    return conn