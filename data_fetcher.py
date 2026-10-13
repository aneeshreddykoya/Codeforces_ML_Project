import sqlite3
import requests
import pandas as pd
import time

DB_NAME = "cf_data.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS submissions
                 (id INTEGER PRIMARY KEY, handle TEXT, problem_id TEXT, rating INTEGER, tags TEXT, verdict TEXT)''')
    conn.commit()
    conn.close()
    print("Database initialized.")

def fetch_user_submissions(handle):
    url = f"https://codeforces.com/api/user.status?handle={handle}"
    try:
        response = requests.get(url).json()
    except Exception as e:
        print(f"API request failed: {e}")
        return
    
    if response['status'] != 'OK':
        print(f"Error fetching data for {handle}")
        return
        
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    count = 0
    for sub in response['result']:
        try:
            problem = sub['problem']
            p_id = str(problem.get('contestId', '')) + problem.get('index', '')
            rating = problem.get('rating', 0)
            tags = ",".join(problem.get('tags', []))
            verdict = sub.get('verdict', '')
            
            c.execute("INSERT OR IGNORE INTO submissions (id, handle, problem_id, rating, tags, verdict) VALUES (?, ?, ?, ?, ?, ?)",
                      (sub['id'], handle, p_id, rating, tags, verdict))
            count += 1
        except Exception as e:
            continue
            
    conn.commit()
    conn.close()
    print(f"Successfully saved {count} submissions for {handle} to SQLite.")

if __name__ == "__main__":
    init_db()
    # Uncomment below to test fetching
    fetch_user_submissions("tourist")
