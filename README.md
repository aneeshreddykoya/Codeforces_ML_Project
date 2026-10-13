# Competitive Programming ML Recommender & Predictor

This is the starter codebase combining Python, C++, FastAPI, and Scikit-Learn to build the data science portfolio project.

## Local Setup (Windows 11 / VS Code)

### 1. Install Python Dependencies
Open your VS Code terminal and install the required data science and backend libraries:
```bash
pip install -r requirements.txt
```

### 2. Compile the C++ Engine (Optional but Recommended)
To enable the high-performance memory management and math calculations:
If you have GCC/MinGW installed, compile the `similarity.cpp` file into a Windows Dynamic Link Library (.dll):
```bash
g++ -shared -o similarity.dll similarity.cpp
```
*Note: If you skip this step, the FastAPI backend will automatically fall back to Python execution.*

### 3. Initialize Database & Fetch Data
Run the data scraper to query the Codeforces API and generate your local SQLite database (`cf_data.db`):
```bash
python data_fetcher.py
```
*(Open the file and uncomment `fetch_user_submissions("tourist")` at the bottom to test it).*

### 4. Start the Web Server
Launch the FastAPI backend:
```bash
python -m uvicorn main:app --reload
```

### 5. View Your A/B Testing Results
Open your browser and navigate to:
- **API Dashboard:** `http://localhost:8000/docs`
- **Test the A/B Router:** `http://localhost:8000/recommend/tourist`
- **View Statistical Significance (p-value):** `http://localhost:8000/ab-test/stats`
