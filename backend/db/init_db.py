from .db import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    strategy TEXT,
    avg_latency FLOAT,
    avg_instances FLOAT,
    total_cost FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

conn.commit()
cur.close()
conn.close()

print("Table created successfully!")