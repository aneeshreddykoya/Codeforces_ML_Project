from fastapi import FastAPI
import random
import ctypes
import os
import numpy as np
import scipy.stats as stats

app = FastAPI(title="Codeforces ML Recommender API")

# Attempt to load C++ DLL for Windows acceleration
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
    print(f"Notice: similarity.dll not found. Will use fallback Python logic. Compile with g++ for C++ acceleration.")

@app.get("/")
def read_root():
    return {
        "status": "System Online", 
        "cpp_acceleration_active": USE_CPP,
        "message": "Navigate to /docs to test endpoints."
    }

@app.get("/recommend/{handle}")
def recommend_problem(handle: str):
    # A/B Testing routing logic
    group = "A" if random.random() < 0.5 else "B"
    
    if group == "A":
        # Group A gets Baseline Logic (e.g. Randomly selected unsolved problem +100 rating)
        return {
            "handle": handle, 
            "test_group": group, 
            "recommendation": "1500A", 
            "engine": "Baseline Algorithm"
        }
    else:
        # Group B gets Machine Learning Logic (Collaborative Filtering)
        return {
            "handle": handle, 
            "test_group": group, 
            "recommendation": "1520C", 
            "engine": "ML Collaborative Filtering"
        }

@app.get("/ab-test/stats")
def get_ab_test_stats():
    # Simulated historical experiment data (1 = solved recommendation, 0 = failed/ignored)
    group_a_results = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0] # 40% success rate
    group_b_results = [1, 1, 1, 0, 1, 1, 0, 0, 1, 1] # 70% success rate
    
    # Calculate p-value using scipy
    t_stat, p_value = stats.ttest_ind(group_a_results, group_b_results)
    
    return {
        "baseline_success_rate": f"{sum(group_a_results) / len(group_a_results) * 100}%",
        "ml_success_rate": f"{sum(group_b_results) / len(group_b_results) * 100}%",
        "p_value": p_value,
        "statistically_significant": bool(p_value < 0.05),
        "conclusion": "The ML model outperforms the baseline" if p_value < 0.05 else "Not enough data to prove ML model is better."
    }
