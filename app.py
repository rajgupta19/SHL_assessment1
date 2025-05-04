import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Tuple
import re
import time
import google.generativeai as genai
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide"
)

# Constants
SHL_CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
MAX_RECOMMENDATIONS = 10
GEMINI_API_KEY = "AIzaSyC0efJe7e6emLz5nMD4ZMsfS8mrZ26QQk0"  # Replace with your actual API key

# Evaluation queries from the requirements
EVALUATION_QUERIES = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
    "Here is a JD text, can you recommend some assessment that can help me screen applications. Time limit is less than 30 minutes.",
    "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins."
]

# Initialize Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
except Exception as e:
    logger.warning(f"Failed to initialize Gemini: {str(e)}")
    model = None

# Sample assessment data (since we can't directly scrape SHL's website)
SAMPLE_ASSESSMENTS = [
    {
        'name': 'Verify Interactive',
        'url': 'https://www.shl.com/solutions/products/verify-interactive/',
        'description': 'Interactive coding assessment for software developers',
        'duration': '60 minutes',
        'test_type': 'Technical',
        'remote_support': 'Yes',
        'adaptive_support': 'Yes'
    },
    {
        'name': 'Verify G+',
        'url': 'https://www.shl.com/solutions/products/verify-g-plus/',
        'description': 'General ability test for graduate and professional roles',
        'duration': '45 minutes',
        'test_type': 'Cognitive',
        'remote_support': 'Yes',
        'adaptive_support': 'Yes'
    },
    {
        'name': 'OPQ',
        'url': 'https://www.shl.com/solutions/products/opq/',
        'description': 'Personality questionnaire for workplace behavior assessment',
        'duration': '30 minutes',
        'test_type': 'Personality',
        'remote_support': 'Yes',
        'adaptive_support': 'No'
    },
    {
        'name': 'Verify Numerical Reasoning',
        'url': 'https://www.shl.com/solutions/products/verify-numerical-reasoning/',
        'description': 'Numerical reasoning test for analytical roles',
        'duration': '35 minutes',
        'test_type': 'Cognitive',
        'remote_support': 'Yes',
        'adaptive_support': 'Yes'
    },
    {
        'name': 'Verify Verbal Reasoning',
        'url': 'https://www.shl.com/solutions/products/verify-verbal-reasoning/',
        'description': 'Verbal reasoning test for communication-focused roles',
        'duration': '30 minutes',
        'test_type': 'Cognitive',
        'remote_support': 'Yes',
        'adaptive_support': 'Yes'
    },
    {
        'name': 'Verify Mechanical Comprehension',
        'url': 'https://www.shl.com/solutions/products/verify-mechanical-comprehension/',
        'description': 'Mechanical reasoning test for technical and engineering roles',
        'duration': '25 minutes',
        'test_type': 'Technical',
        'remote_support': 'Yes',
        'adaptive_support': 'No'
    },
    {
        'name': 'Verify Interactive - Java',
        'url': 'https://www.shl.com/solutions/products/verify-interactive-java/',
        'description': 'Java programming assessment for developers',
        'duration': '60 minutes',
        'test_type': 'Technical',
        'remote_support': 'Yes',
        'adaptive_support': 'Yes'
    },
    {
        'name': 'Verify Interactive - Python',
        'url': 'https://www.shl.com/solutions/products/verify-interactive-python/',
        'description': 'Python programming assessment for developers',
        'duration': '60 minutes',
        'test_type': 'Technical',
        'remote_support': 'Yes',
        'adaptive_support': 'Yes'
    },
    {
        'name': 'Verify Interactive - SQL',
        'url': 'https://www.shl.com/solutions/products/verify-interactive-sql/',
        'description': 'SQL programming assessment for database professionals',
        'duration': '45 minutes',
        'test_type': 'Technical',
        'remote_support': 'Yes',
        'adaptive_support': 'Yes'
    },
    {
        'name': 'Verify Interactive - JavaScript',
        'url': 'https://www.shl.com/solutions/products/verify-interactive-javascript/',
        'description': 'JavaScript programming assessment for web developers',
        'duration': '60 minutes',
        'test_type': 'Technical',
        'remote_support': 'Yes',
        'adaptive_support': 'Yes'
    }
]

def calculate_metrics(recommendations: List[Dict], relevant_assessments: List[str], k: int = 3) -> Tuple[float, float]:
    """
    Calculate Mean Recall@K and MAP@K for the recommendations.
    """
    if not recommendations or not relevant_assessments:
        return 0.0, 0.0
    
    # Get top K recommendations
    top_k = recommendations[:k]
    
    # Calculate Recall@K
    relevant_found = sum(1 for rec in top_k if rec['name'] in relevant_assessments)
    recall_at_k = relevant_found / len(relevant_assessments) if relevant_assessments else 0
    
    # Calculate MAP@K
    precision_sum = 0
    relevant_count = 0
    
    for i, rec in enumerate(top_k):
        if rec['name'] in relevant_assessments:
            relevant_count += 1
            precision_at_k = relevant_count / (i + 1)
            precision_sum += precision_at_k
    
    map_at_k = precision_sum / len(relevant_assessments) if relevant_assessments else 0
    
    return recall_at_k, map_at_k

def get_llm_recommendations(query: str, assessments_df: pd.DataFrame) -> List[str]:
    """
    Use Gemini to identify relevant assessments based on the query.
    """
    if not model:
        return []
    
    try:
        # Create a prompt for the LLM
        prompt = f"""
        Given the following job description/query:
        "{query}"
        
        And these available assessments:
        {assessments_df[['name', 'description', 'duration', 'test_type']].to_string()}
        
        Please identify the most relevant assessments (up to 3) that would be suitable for this role.
        Return only the assessment names, one per line.
        """
        
        response = model.generate_content(prompt)
        recommended_names = [name.strip() for name in response.text.split('\n') if name.strip()]
        return recommended_names
    except Exception as e:
        logger.error(f"Error getting LLM recommendations: {str(e)}")
        return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_assessment_data() -> pd.DataFrame:
    """
    Get assessment data from SHL catalog by scraping the product catalog page.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(SHL_CATALOG_URL, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        assessments = []
        
        # Find all assessment links
        assessment_links = soup.find_all('a', href=re.compile(r'/products/product-catalog/view/.*'))
        
        for link in assessment_links:
            # Get the assessment URL
            assessment_url = f"https://www.shl.com{link['href']}"
            
            # Get the assessment name
            name = link.text.strip()
            
            # Try to get additional details from the parent elements
            parent_div = link.find_parent('div', class_=lambda x: x and 'product-item' in x.lower())
            
            # Extract description if available
            description = ""
            desc_elem = parent_div.find('p') if parent_div else None
            if desc_elem:
                description = desc_elem.text.strip()
            
            # Extract duration if available
            duration = "Not specified"
            duration_elem = parent_div.find('span', class_=lambda x: x and 'duration' in x.lower()) if parent_div else None
            if duration_elem:
                duration = duration_elem.text.strip()
            
            # Extract test type if available
            test_type = "Not specified"
            type_elem = parent_div.find('span', class_=lambda x: x and 'type' in x.lower()) if parent_div else None
            if type_elem:
                test_type = type_elem.text.strip()
            
            # Determine remote and adaptive support based on description
            remote_support = 'Yes' if 'remote' in description.lower() else 'No'
            adaptive_support = 'Yes' if 'adaptive' in description.lower() or 'irt' in description.lower() else 'No'
            
            assessment = {
                'name': name,
                'url': assessment_url,
                'description': description,
                'duration': duration,
                'test_type': test_type,
                'remote_support': remote_support,
                'adaptive_support': adaptive_support
            }
            assessments.append(assessment)
        
        if not assessments:
            logger.warning("No assessments found. Using sample data instead.")
            return pd.DataFrame(SAMPLE_ASSESSMENTS)
        
        return pd.DataFrame(assessments)
    except Exception as e:
        logger.warning(f"Error scraping SHL catalog: {str(e)}. Using sample data instead.")
        return pd.DataFrame(SAMPLE_ASSESSMENTS)

def get_recommendations(query: str, df: pd.DataFrame, max_recommendations: int = MAX_RECOMMENDATIONS) -> List[Dict]:
    """
    Get assessment recommendations based on the query using both TF-IDF and LLM.
    """
    if df.empty:
        return []
    
    # Get LLM recommendations
    llm_recommendations = get_llm_recommendations(query, df)
    
    # Combine relevant text fields for similarity calculation
    df['combined_text'] = df['name'] + ' ' + df['description']
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top recommendations
    top_indices = similarity_scores.argsort()[-max_recommendations:][::-1]
    
    recommendations = []
    for idx in top_indices:
        if similarity_scores[idx] > 0:  # Only include if there's some similarity
            assessment = df.iloc[idx].to_dict()
            # Boost score if recommended by LLM
            if assessment['name'] in llm_recommendations:
                similarity_scores[idx] *= 1.5
            
            recommendations.append({
                'name': assessment['name'],
                'url': assessment['url'],
                'remote_support': assessment['remote_support'],
                'adaptive_support': assessment['adaptive_support'],
                'duration': assessment['duration'],
                'test_type': assessment['test_type'],
                'similarity_score': float(similarity_scores[idx])
            })
    
    # Sort by similarity score
    recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
    return recommendations[:max_recommendations]

def evaluate_system():
    """
    Evaluate the system using the benchmark queries.
    """
    df = get_assessment_data()
    total_recall = 0
    total_map = 0
    
    for query in EVALUATION_QUERIES:
        recommendations = get_recommendations(query, df)
        llm_recommendations = get_llm_recommendations(query, df)
        
        recall, map_score = calculate_metrics(recommendations, llm_recommendations)
        total_recall += recall
        total_map += map_score
    
    mean_recall = total_recall / len(EVALUATION_QUERIES)
    mean_map = total_map / len(EVALUATION_QUERIES)
    
    return mean_recall, mean_map

def main():
    st.title("🎯 SHL Assessment Recommender")
    st.write("Enter a job description or natural language query to get relevant assessment recommendations.")
    
    # Sidebar for evaluation metrics
    with st.sidebar:
        st.header("System Evaluation")
        if st.button("Run Evaluation"):
            with st.spinner("Evaluating system..."):
                mean_recall, mean_map = evaluate_system()
                st.metric("Mean Recall@3", f"{mean_recall:.2%}")
                st.metric("Mean MAP@3", f"{mean_map:.2%}")
    
    # Input section
    query = st.text_area(
        "Enter your job description or query:",
        height=150,
        placeholder="Example: I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
    )
    
    if st.button("Get Recommendations"):
        if not query:
            st.warning("Please enter a query to get recommendations.")
            return
        
        with st.spinner("Fetching recommendations..."):
            # Get assessment data
            df = get_assessment_data()
            
            if df.empty:
                st.error("Unable to fetch assessment data. Please try again later.")
                return
            
            # Get recommendations
            recommendations = get_recommendations(query, df)
            
            if not recommendations:
                st.warning("No relevant assessments found. Try modifying your query.")
                return
            
            # Display recommendations
            st.subheader("Recommended Assessments")
            
            # Create a DataFrame for display with only name and url
            display_df = pd.DataFrame(recommendations)[["name", "url"]]
            
            # Format the DataFrame for display
            st.dataframe(
                display_df,
                column_config={
                    "name": "Assessment Name",
                    "url": st.column_config.LinkColumn("URL")
                },
                hide_index=True
            )
            
            # Display evaluation metrics for this query
            llm_recommendations = get_llm_recommendations(query, df)
            recall, map_score = calculate_metrics(recommendations, llm_recommendations)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Recall@3", f"{recall:.2%}")
            with col2:
                st.metric("MAP@3", f"{map_score:.2%}")

if __name__ == "__main__":
    main() 