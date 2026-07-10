import sqlite3

conn = sqlite3.connect('sqa_agent.db')
cursor = conn.cursor()
cursor.execute("UPDATE users SET role='admin' WHERE email='admin2@sqa.com'")
conn.commit()
print("Rows updated:", cursor.rowcount)
conn.close()