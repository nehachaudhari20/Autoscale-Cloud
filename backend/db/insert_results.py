from backend.db.db import get_connection

def insert_result(strategy, latency, instances, cost):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO experiments (strategy, avg_latency, avg_instances, total_cost)
        VALUES (%s, %s, %s, %s)
    """, (strategy, latency, instances, cost))

    conn.commit()
    cur.close()
    conn.close()