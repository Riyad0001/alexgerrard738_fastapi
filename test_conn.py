import os
import psycopg2
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

db_url = os.getenv("DATABASE_URL")
print(f"Connecting to database: {db_url}")

try:
    conn = psycopg2.connect(db_url, connect_timeout=5)
    cursor = conn.cursor()
    cursor.execute("SELECT 1;")
    result = cursor.fetchone()
    print("\n===============================")
    print(f"SUCCESS! Connected successfully.")
    print(f"Query Result: {result}")
    print("===============================\n")
    conn.close()
except Exception as e:
    print("\n===============================")
    print(f"CONNECTION FAILED!")
    print(f"Error: {e}")
    print("===============================\n")
