import sqlite3
from datetime import datetime

DB_NAME = "ecotogether.db"

def connect_db():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def create_table():
    conn = connect_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            points INTEGER DEFAULT 0
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            points INTEGER,
            reason TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()

def get_user(username):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT points FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0

def update_user(username, add_points, reason="AI submit"):
    conn = connect_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO users (username, points)
        VALUES (?, ?)
        ON CONFLICT(username)
        DO UPDATE SET points = points + ?
    """, (username, add_points, add_points))

    cur.execute("""
        INSERT INTO transactions (username, points, reason, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        username,
        add_points,
        reason,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

    return get_user(username)


