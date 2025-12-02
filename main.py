from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Resume Parser",
    description="AI-powered resume analysis for Senior Python Developer positions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models for Input/Output Validation
class ResumeInput(BaseModel):
    """input model for resume text"""
    resume_text: str = Field(..., min_length=50, description="raw resume text to parse")
    
    @field_validator('resume_text')
    def validate_resume_text(cls, v):
        if len(v.strip()) < 50:
            raise ValueError('Resume text must be at least 50 characters')
        return v.strip()


class ParsedResume(BaseModel):
    """Output model for parsed resume data"""
    candidate_name: str = Field(..., description="Get candidate name")
    tech_stack: List[str] = Field(..., description="List of technical skill and technologies")
    years_of_experience: float = Field(..., ge=0, description="Total Experience")
    fit_score: float = Field(..., ge=0, le=10, description="Get the score from 0 to 10")
    fit_explanation: str = Field(..., description="Explanation of the score")
    matching_skills: List[str] = Field(..., description="Skills matching the job requirement")
    missing_skills: List[str] = Field(default=[], description="Important skills not found in resume to tell the candidate")


# Job Requirements (Hardcoded for Senior Python Developer)
SENIOR_PYTHON_REQUIREMENTS = {
    "must_have_skills": [
        "Python", "FastAPI", "Flask", "Django", 
        "REST API", "SQL", "Git"
    ],
    "preferred_skills": [
        "AI", "Machine Learning", "TensorFlow", "PyTorch",
        "LangChain", "OpenAI", "Docker", "Kubernetes",
        "AWS", "Azure", "GCP", "Redis", "PostgreSQL",
        "MongoDB", "CI/CD", "Testing", "pytest"
    ],
    "min_years": 1
}


def initialize_llm():
    """initialize Groq LLM with API key from environment"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    return ChatGroq(
        temperature=0,
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile"  # Fast and accurate model
    )


def extract_resume_data(resume_text: str, llm) -> dict:
    """
    Use LLM to extract structured data from resume
    Returns: dict with name, tech_stack, and work_history
    """
    
    prompt_template = """
You are an expert resume parser. Extract the following information from the resume text and return ONLY a valid JSON object with no additional text, preamble, or explanation.

Resume Text:
{resume_text}

Extract and return a JSON object with this exact structure:
{{
    "candidate_name": "Full name of the candidate",
    "tech_stack": ["skill1", "skill2", "skill3"],
    "work_history": [
        {{
            "company": "Company Name",
            "position": "Job Title",
            "start_date": "YYYY",
            "end_date": "YYYY or Present"
        }}
    ]
}}

Important Instructions:
1. Extract ALL technical skills, frameworks, languages, tools, and technologies mentioned
2. For work history, extract ALL positions with date ranges
3. Convert all dates to year format (YYYY)
4. If end date is current, use "Present"
5. Return ONLY the JSON object, no other text
6. Ensure the JSON is valid and properly formatted

JSON Output:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["resume_text"]
    )
    
    try:
        # Get response from LLM
        chain = prompt | llm
        response = chain.invoke({"resume_text": resume_text})
        
        # Extract JSON from response
        content = response.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("json"):
            content = content.replace("json", "").replace("", "").strip()
        elif content.startswith(""):
            content = content.replace("```", "").strip()
        
        # Parse JSON
        parsed_data = json.loads(content)
        
        return parsed_data
    
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse LLM response as JSON: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting resume data: {str(e)}"
        )


def calculate_years_of_experience(work_history: List[dict]) -> float:
    """
    Calculate total years of experience from work history
    Handles overlapping periods and current employment
    """
    current_year = datetime.now().year
    total_months = 0
    
    for job in work_history:
        try:
            start_year = int(job.get("start_date", "0"))
            end_date = job.get("end_date", "Present")
            
            if end_date.lower() == "present":
                end_year = current_year
            else:
                end_year = int(end_date)
            
            if start_year > 0 and end_year >= start_year:
                years = end_year - start_year
                # If job is current or ended this year, add partial year
                if end_date.lower() == "present":
                    months = (current_year - start_year) * 12
                else:
                    months = years * 12
                total_months += months
        
        except (ValueError, TypeError):
            continue
    
    return round(total_months / 12, 1)


def calculate_fit_score(tech_stack: List[str], years_exp: float) -> tuple:
    """
    Calculate fit score (0-10) based on skills and experience
    Returns: (score, explanation, matching_skills, missing_skills)
    """
    must_have = set(s.lower() for s in SENIOR_PYTHON_REQUIREMENTS["must_have_skills"])
    preferred = set(s.lower() for s in SENIOR_PYTHON_REQUIREMENTS["preferred_skills"])
    candidate_skills = set(s.lower() for s in tech_stack)
    
    # Calculate skill matches
    must_have_matches = candidate_skills & must_have
    preferred_matches = candidate_skills & preferred
    
    # Calculate score components
    must_have_score = (len(must_have_matches) / len(must_have)) * 5  # 50% weight
    preferred_score = (len(preferred_matches) / len(preferred)) * 3  # 30% weight
    
    # Experience score (20% weight)
    exp_score = min(years_exp / SENIOR_PYTHON_REQUIREMENTS["min_years"], 1.0) * 2
    
    total_score = must_have_score + preferred_score + exp_score
    total_score = round(min(total_score, 10), 1)
    
    # Find matching and missing skills
    matching_skills = list(must_have_matches | preferred_matches)
    matching_skills = [s.title() for s in matching_skills]
    
    missing_must_have = must_have - candidate_skills
    missing_preferred = preferred - candidate_skills
    important_missing = list((missing_must_have | missing_preferred) & 
                           set(['python', 'fastapi', 'docker', 'sql', 'redis', 'postgresql']))
    missing_skills = [s.title() for s in important_missing]
    
    # Generate explanation
    if total_score >= 8:
        level = "Excellent"
    elif total_score >= 6:
        level = "Good"
    elif total_score >= 4:
        level = "Fair"
    else:
        level = "Limited"
    
    explanation = (
        f"{level} match for Senior Python Developer role. "
        f"{'Strong' if years_exp >= 5 else 'Moderate'} Python background with {years_exp} years of experience. "
        f"Found {len(must_have_matches)}/{len(must_have)} must-have skills and "
        f"{len(preferred_matches)}/{len(preferred)} preferred skills. "
    )
    
    if total_score >= 7:
        explanation += "Highly recommended for interview."
    elif total_score >= 5:
        explanation += "Recommended for further review."
    else:
        explanation += "May need additional skills development."
    
    return total_score, explanation, matching_skills, missing_skills


@app.get("/")
async def root():
    """Serve the UI homepage"""
    return FileResponse("static/index.html")


@app.post("/api/parse-resume", response_model=ParsedResume)
async def parse_resume(resume: ResumeInput):
    """
    Parse resume and return structured analysis
    
    - *resume_text*: Raw resume text (minimum 50 characters)
    
    Returns structured JSON with candidate info, experience, and fit score
    """
    try:
        # Initialize LLM
        llm = initialize_llm()
        
        # Extract data using LLM
        extracted_data = extract_resume_data(resume.resume_text, llm)
        
        # Calculate years of experience
        years_exp = calculate_years_of_experience(
            extracted_data.get("work_history", [])
        )
        
        # Calculate fit score
        fit_score, explanation, matching_skills, missing_skills = calculate_fit_score(
            extracted_data.get("tech_stack", []),
            years_exp
        )
        
        # Prepare response
        result = ParsedResume(
            candidate_name=extracted_data.get("candidate_name", "Unknown"),
            tech_stack=extracted_data.get("tech_stack", []),
            years_of_experience=years_exp,
            fit_score=fit_score,
            fit_explanation=explanation,
            matching_skills=matching_skills,
            missing_skills=missing_skills
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing resume: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "groq_api_configured": bool(os.getenv("GROQ_API_KEY"))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)