from backend.db.db import get_connection

def fetch_results():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, strategy, avg_latency, avg_instances, total_cost, created_at
        FROM experiments
        ORDER BY created_at DESC
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows