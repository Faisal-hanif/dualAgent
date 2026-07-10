import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "sqa_agent.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    print("📦 Creating tables...")

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            url TEXT NOT NULL,
            score INTEGER,
            load_time REAL,
            total_links INTEGER,
            working_links INTEGER,
            broken_links INTEGER,
            broken_links_list TEXT,
            technologies TEXT,
            trust_score INTEGER,
            excitement_score INTEGER,
            professionalism_score INTEGER,
            tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            test_result_id INTEGER,
            url TEXT NOT NULL,
            score INTEGER,
            tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            test_result_id INTEGER,
            url TEXT NOT NULL,
            score INTEGER,
            broken_count INTEGER,
            suggestion TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

    print("✅ ALL TABLES CREATED SUCCESSFULLY!")