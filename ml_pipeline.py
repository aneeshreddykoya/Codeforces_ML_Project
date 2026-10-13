import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DB_NAME = "cf_data.db"

def build_user_problem_matrix():
    conn = sqlite3.connect(DB_NAME)
    # Fetch only successful solves to build the recommendation matrix
    df = pd.read_sql_query("SELECT handle, problem_id, verdict FROM submissions WHERE verdict = 'OK'", conn)
    conn.close()
    
    if df.empty:
        print("Database is empty. Run data_fetcher.py first.")
        return pd.DataFrame()
        
    df['solved'] = 1
    matrix = df.drop_duplicates(subset=['handle', 'problem_id']).pivot(index='handle', columns='problem_id', values='solved').fillna(0)
    print(f"Built User-Problem matrix with shape: {matrix.shape}")
    return matrix

def train_rating_predictor():
    print("Initializing RandomForestRegressor for rating prediction...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    # In a full implementation, you would merge user metrics (e.g. solve variance) with contest ratings here
    # model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    matrix = build_user_problem_matrix()
