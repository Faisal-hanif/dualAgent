import sqlite3
import os

import os

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_b1q0RzWZrcIl@ep-still-dream-atfk5a1y.c-9.us-east-1.aws.neon.tech/neondb",
)


class DatabaseConnection:
    def __init__(self, connection):
        self._connection = connection

    def cursor(self):
        return self._connection.cursor(cursor_factory=RealDictCursor)

    def commit(self):
        return self._connection.commit()

    def rollback(self):
        return self._connection.rollback()

    def close(self):
        return self._connection.close()


def get_db():
    return DatabaseConnection(psycopg2.connect(DATABASE_URL))


def init_db():
    conn = get_db()
    cursor = conn.cursor()

    print("📦 Creating tables...")

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS test_results (
            id SERIAL PRIMARY KEY,
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
        '''
    )

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS test_history (
            id SERIAL PRIMARY KEY,
            user_id INTEGER,
            test_result_id INTEGER,
            url TEXT NOT NULL,
            score INTEGER,
            tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS ai_suggestions (
            id SERIAL PRIMARY KEY,
            test_result_id INTEGER,
            url TEXT NOT NULL,
            score INTEGER,
            broken_count INTEGER,
            suggestion TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )

    conn.commit()
    conn.close()

    print("✅ ALL TABLES CREATED SUCCESSFULLY!")

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