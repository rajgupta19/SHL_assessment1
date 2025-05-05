from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on job descriptions",
    version="1.0.0"
)

# Sample assessment data in the required format
ASSESSMENTS = [
    {
        "url": "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
        "adaptive_support": "No",
        "description": "Multi-choice test that measures the knowledge of Python programming, databases, modules and library.",
        "duration": 11,
        "remote_support": "Yes",
        "test_type": ["Knowledge & Skills"]
    },
    {
        "url": "https://www.shl.com/solutions/products/product-catalog/view/technology-professional-8-0-job-focused-assessment/",
        "adaptive_support": "No",
        "description": "The Technology Job Focused Assessment assesses key behavioral attributes required for success in fast-paced technology roles.",
        "duration": 16,
        "remote_support": "Yes",
        "test_type": ["Competencies", "Personality & Behaviour"]
    },
    {
        "url": "https://www.shl.com/solutions/products/product-catalog/view/account-manager/",
        "adaptive_support": "Yes",
        "description": "Assessment for mid-level leadership positions managing client accounts, project plans, and internal coordination.",
        "duration": 45,
        "remote_support": "Yes",
        "test_type": ["Leadership", "Client Management"]
    },
    {
        "url": "https://www.shl.com/solutions/products/product-catalog/view/leadership/",
        "adaptive_support": "Yes",
        "description": "Measures leadership capabilities, decision-making, and team management skills.",
        "duration": 60,
        "remote_support": "No",
        "test_type": ["Leadership"]
    },
    {
        "url": "https://www.shl.com/solutions/products/product-catalog/view/client-management/",
        "adaptive_support": "No",
        "description": "Assesses client relationship management, communication, and problem-solving skills.",
        "duration": 40,
        "remote_support": "Yes",
        "test_type": ["Client Management"]
    }
]

class QueryRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[Assessment]

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    Returns:
        dict: Status of the API
    """
    return {"status": "healthy"}

def get_recommendations(query: str, max_recommendations: int = 10) -> List[dict]:
    """
    Get assessment recommendations based on query using TF-IDF and cosine similarity.
    Args:
        query (str): The job description or natural language query
        max_recommendations (int): Maximum number of recommendations to return
    Returns:
        List[dict]: List of recommended assessments
    """
    assessment_texts = [
        f"{a['description']} {' '.join(a['test_type'])}" for a in ASSESSMENTS
    ]
    corpus = assessment_texts + [query]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = tfidf_matrix[-1]
    assessment_vectors = tfidf_matrix[:-1]
    similarity_scores = cosine_similarity(query_vector, assessment_vectors).flatten()
    top_indices = similarity_scores.argsort()[::-1]
    recommendations = []
    for idx in top_indices:
        if similarity_scores[idx] > 0 and len(recommendations) < max_recommendations:
            recommendations.append(ASSESSMENTS[idx])
    if not recommendations:
        # Always return at least one if nothing matches
        recommendations.append(ASSESSMENTS[0])
    return recommendations

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: QueryRequest):
    """
    Process a job description or natural language query and return recommended assessments.
    Args:
        request (QueryRequest): The request containing the query
    Returns:
        RecommendationResponse: The response containing recommended assessments
    """
    try:
        recommendations = get_recommendations(request.query)
        return {"recommended_assessments": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 