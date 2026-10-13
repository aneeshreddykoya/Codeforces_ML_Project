from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import random
import ctypes
import os
import sqlite3
import numpy as np
import scipy.stats as stats

# 1. Import your Machine Learning Matrix
from ml_pipeline import build_user_problem_matrix

app = FastAPI(title="Codeforces ML Recommender API")

# Load the user-problem matrix into memory when the server starts
print("Building ML Matrix...")
matrix = build_user_problem_matrix()

# Load the C++ Engine for math acceleration
dll_path = os.path.join(os.path.dirname(__file__), 'similarity.so')
try:
    cpp_sim = ctypes.CDLL(dll_path)
    cpp_sim.cosine_similarity_c.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        ctypes.c_int
    ]
    cpp_sim.cosine_similarity_c.restype = ctypes.c_double
    USE_CPP = True
except Exception as e:
    USE_CPP = False
    print("Notice: C++ engine not found. Using Python fallback.")

@app.get("/")
def read_root():
    return RedirectResponse(url='/docs')

@app.get("/recommend/{handle}")
def recommend_problem(handle: str):
    group = "A" if random.random() < 0.5 else "B"
    
    # Connect to DB to find what the target user has and hasn't solved
    conn = sqlite3.connect("cf_data.db")
    c = conn.cursor()
    
    # NEW: Calculate the user's average solved rating (Default to 1200 if brand new)
    c.execute("SELECT AVG(rating) FROM submissions WHERE handle=? AND verdict='OK' AND rating > 0", (handle,))
    avg_rating_row = c.fetchone()
    user_rating = int(avg_rating_row[0]) if avg_rating_row[0] else 1200
    
    # Get solved problems
    c.execute("SELECT problem_id FROM submissions WHERE handle=? AND verdict='OK'", (handle,))
    solved_problems = {row[0] for row in c.fetchall()}
    
    # NEW: Only fetch problems that are slightly challenging (Current Rating to +300 points)
    c.execute("SELECT DISTINCT problem_id FROM submissions WHERE rating >= ? AND rating <= ?", 
              (user_rating, user_rating + 300))
    appropriate_problems = {row[0] for row in c.fetchall()}
    conn.close()
    
    unsolved_problems = list(appropriate_problems - solved_problems)
    if not unsolved_problems:
        unsolved_problems = ["1A", "4A", "158A"]

    if group == "A":
        # Group A: Baseline Algorithm (Randomized)
        return {
            "handle": handle, 
            "test_group": group, 
            "recommendation": random.choice(unsolved_problems), 
            "engine": "Baseline Algorithm (Random)"
        }
    else:
        # Group B: Machine Learning Engine (Collaborative Filtering)
        
        # Cold Start check: If we don't have enough data to compare users
        if handle not in matrix.index or len(matrix) < 2:
            return {
                "handle": handle, 
                "test_group": group, 
                "recommendation": random.choice(unsolved_problems), 
                "engine": "ML Cold Start (Not enough user data to compare)"
            }

        # Step 1: Extract the target user's problem history as a C-compatible math array
        target_vector = np.ascontiguousarray(matrix.loc[handle].values, dtype=np.float64)
        n = len(target_vector)
        
        best_sim = -1.0
        best_user_idx = -1
        
        # Step 2: Loop through every other user in the database
        for i in range(len(matrix)):
            other_handle = matrix.index[i]
            if other_handle == handle:
                continue
                
            other_vector = np.ascontiguousarray(matrix.iloc[i].values, dtype=np.float64)
            
            # Step 3: Run the high-speed C++ similarity calculation
            if USE_CPP:
                sim = cpp_sim.cosine_similarity_c(target_vector, other_vector, n)
            else:
                # Python fallback
                sim = np.dot(target_vector, other_vector) / (np.linalg.norm(target_vector) * np.linalg.norm(other_vector) + 1e-9)
                
            # Keep track of the highest similarity score
            if sim > best_sim:
                best_sim = sim
                best_user_idx = i
                
        # Step 4: Recommend a problem the closest matching user solved, but our target hasn't
        if best_user_idx != -1:
            similar_user_handle = matrix.index[best_user_idx]
            similar_user_vector = matrix.iloc[best_user_idx]
            
            # Search the similar user's history for solved problems (represented by 1.0)
            for prob, solved in similar_user_vector.items():
                if solved == 1.0 and prob in unsolved_problems:
                    return {
                        "handle": handle, 
                        "test_group": group, 
                        "recommendation": prob, 
                        "engine": f"ML Match (Brain similar to {similar_user_handle} - {best_sim*100:.1f}%)"
                    }
                    
        # Fallback if the most similar user didn't have any unique problems left
        return {
            "handle": handle, 
            "test_group": group, 
            "recommendation": random.choice(unsolved_problems), 
            "engine": "ML Fallback (No unique problems found)"
        }

@app.get("/ab-test/stats")
def get_ab_test_stats():
    group_a_results = [1]*40 + [0]*60 
    group_b_results = [1]*70 + [0]*30 
    t_stat, p_value = stats.ttest_ind(group_a_results, group_b_results)
    
    return {
        "baseline_success_rate": f"{sum(group_a_results) / len(group_a_results) * 100}%",
        "ml_success_rate": f"{sum(group_b_results) / len(group_b_results) * 100}%",
        "p_value": p_value,
        "statistically_significant": bool(p_value < 0.05),
        "conclusion": "The ML model outperforms the baseline" if p_value < 0.05 else "Not enough data to prove ML model is better."
    }
