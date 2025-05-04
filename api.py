from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from app import get_assessment_data, get_recommendations

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on job descriptions",
    version="1.0.0"
)

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
    """
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_assessment_recommendations(request: QueryRequest):
    """
    Get assessment recommendations based on the query.
    """
    try:
        # Get assessment data
        df = get_assessment_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="Unable to fetch assessment data")
        # Get recommendations
        recommendations = get_recommendations(request.query, df)
        if not recommendations:
            raise HTTPException(status_code=404, detail="No relevant assessments found")
        # Prepare response in required format
        recs = []
        for rec in recommendations:
            # Parse duration to integer (extract number from string)
            duration_val = 0
            if 'duration' in rec and rec['duration']:
                import re
                match = re.search(r'(\d+)', str(rec['duration']))
                if match:
                    duration_val = int(match.group(1))
            # Parse test_type as array of strings
            test_type_val = []
            if 'test_type' in rec and rec['test_type']:
                # If comma or & separated, split
                if isinstance(rec['test_type'], list):
                    test_type_val = rec['test_type']
                else:
                    test_type_val = [t.strip() for t in str(rec['test_type']).replace('[','').replace(']','').replace('"','').replace("'",'').replace(';',',').split(',') if t.strip()]
            recs.append({
                "url": rec.get("url", ""),
                "adaptive_support": rec.get("adaptive_support", "No"),
                "description": rec.get("description", ""),
                "duration": duration_val,
                "remote_support": rec.get("remote_support", "No"),
                "test_type": test_type_val
            })
        return {"recommended_assessments": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 