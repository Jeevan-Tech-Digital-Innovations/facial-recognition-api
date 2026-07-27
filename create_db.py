import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

host = "168.231.103.17"
port = 5432
user = "staging_admin"
password = "JGJGAjg234jGJAGDJLAu"
dbname = "biometric_staging"

conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname="postgres")
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cur = conn.cursor()

cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
exists = cur.fetchone()

if exists:
    print(f"Database '{dbname}' already exists.")
else:
    cur.execute(f"CREATE DATABASE {dbname}")
    print(f"Database '{dbname}' created successfully.")

cur.close()
conn.close()
