# SHL Assessment Recommender API Documentation

## Overview
This API provides endpoints for recommending SHL assessments based on job descriptions or natural language queries. The API implements the exact requirements specified in Appendix 2 of the assignment.

## Setup

1. Install dependencies:
```bash
pip install fastapi uvicorn scikit-learn numpy
```

2. Run the API server:
```bash
python text_query_api.py
```

The API will be available at: http://localhost:8000

## API Endpoints

### 1. Health Check Endpoint
- **Endpoint**: `GET /health`
- **Purpose**: Verify the API is running
- **Response**: 
```json
{
    "status": "healthy"
}
```

### 2. Assessment Recommendation Endpoint
- **Endpoint**: `POST /recommend`
- **Purpose**: Accept a job description or natural language query and return recommended assessments
- **Request Body**:
```json
{
    "query": "The Account Manager solution is an assessment used for job candidates applying to mid-level leadership positions..."
}
```
- **Response Format**:
```json
{
    "recommended_assessments": [
        {
            "id": "AM001",
            "name": "Account Manager Assessment",
            "description": "Evaluates skills for managing client accounts, project coordination, and client communication",
            "duration": 45,
            "type": "Leadership",
            "skills": ["Client Management", "Project Coordination", "Communication", "Leadership"],
            "confidence": 0.95
        },
        {
            "id": "CM001",
            "name": "Client Management Assessment",
            "description": "Assesses client relationship management, communication, and problem-solving skills",
            "duration": 40,
            "type": "Client Management",
            "skills": ["Client Relations", "Communication", "Problem Solving", "Negotiation"],
            "confidence": 0.90
        }
    ]
}
```

## Response Fields Explanation

For the `/recommend` endpoint response:
- `recommended_assessments`: List of recommended assessments (1-10 assessments)
  - `id`: Unique identifier for the assessment
  - `name`: Name of the assessment
  - `description`: Detailed description of the assessment
  - `duration`: Duration in minutes
  - `type`: Type of assessment
  - `skills`: List of skills being assessed
  - `confidence`: Confidence score (0-1) indicating relevance to the query

## Testing Your API

Before submitting your API, verify that:
1. Both endpoints are functioning correctly
2. The response formats match exactly what is specified above
3. The `/recommend` endpoint returns between 1-10 assessments
4. All responses are in JSON format
5. Proper HTTP status codes are used

## Example Usage

### Using curl:
```bash
# Health Check
curl http://localhost:8000/health

# Get Recommendations
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{
         "query": "The Account Manager solution is an assessment used for job candidates applying to mid-level leadership positions..."
     }'
```

### Using Python:
```python
import requests

# Health Check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Get Recommendations
query = """
The Account Manager solution is an assessment used for job candidates applying to mid-level leadership positions...
"""

response = requests.post(
    "http://localhost:8000/recommend",
    json={"query": query}
)
print(response.json())
``` 