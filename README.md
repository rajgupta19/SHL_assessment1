# SHL Assessment Recommender

## Quickstart

### 1. Clone the repository
```bash
git clone <repository-url>
cd shl-recommender
```

### 2. Create and activate a virtual environment
- **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **On Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the API server
```bash
python api.py
```
- The API will be available at: http://localhost:8000
- Test the health endpoint: http://localhost:8000/health
- Test the recommendation endpoint using curl or Postman:
  ```bash
  curl -X POST "http://localhost:8000/recommend" \
       -H "Content-Type: application/json" \
       -d '{"query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."}'
  ```

---

An intelligent recommendation system that helps hiring managers find the most relevant SHL assessments based on job descriptions or natural language queries.

## Features

- Natural language query processing
- Intelligent assessment recommendations using TF-IDF and cosine similarity
- Clean and intuitive user interface
- Displays key assessment attributes:
  - Assessment name and URL
  - Remote Testing Support
  - Adaptive/IRT Support
  - Duration
  - Test type

## Usage (Streamlit UI)

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter your job description or query in the text area

4. Click "Get Recommendations" to see the relevant assessments

## Technical Details

- Built with Streamlit for the user interface
- Uses scikit-learn for text processing and similarity calculations
- Implements web scraping to fetch assessment data from SHL's catalog
- Employs TF-IDF vectorization and cosine similarity for recommendations

## Evaluation Metrics

The system is evaluated using:
- Mean Recall@K
- Mean Average Precision @K (MAP@K)

## API Endpoints

The application provides two main endpoints:
1. Health Check Endpoint
2. Assessment Recommendation Endpoint

## License

[Your chosen license] 