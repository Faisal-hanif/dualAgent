import os

import psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cursor = conn.cursor()
cursor.execute("UPDATE users SET role='admin' WHERE email=%s", ("admin2@sqa.com",))
conn.commit()
print("Rows updated:", cursor.rowcount)
conn.close()