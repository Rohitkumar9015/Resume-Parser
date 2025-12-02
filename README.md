#  Smart Resume Parser
##  Features

### Core Functionality
- Candidate Extraction*: Get the candidate name and complete tech stack
- Experience Calculation*: calculates total years of experience from date ranges
- Fit Scoring*: Get resume based score 0-10 fit and skills match with job requirements
- Detailed Analysis*: Provides matching skills, telling the missing skills, and fit explanation

### Bonus Features Implemented
- *Input Validation*: Pydantic models for strict request/response validation
- *Simple UI*: Clean and easy web interface for use everyone
- *API Documentation*: Auto-generated Swagger UI at /docs

##  Architecture


smart-resume-parser/
├── main.py                 # FastAPI application with business logic
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── .env                   # Environment variables (create from .env.example)
│── index.html        # Web UI
└── README.md             # This file


## Technology Stack

- *Framework*: FastAPI 0.109.0
- *LLM Framework*: LangChain 0.1.5
- *LLM Provider*: Groq Cloud (llama-3.3-70b-versatile)
- *Validation*: Pydantic 2.5.3
- *Containerization*: Docker & Docker Compose

## Prerequisites

- Python 3.11+
- Docker 
- Groq Cloud API Key 

## Quick Start

###  Local Development

1. *Install dependencies*
bash
pip install -r requirements.txt


2. *Configure environment*

# Edit .env with your GROQ_API_KEY


3. *Run the server*
bash
python main.py
# Or use uvicorn directly:
uvicorn main:app --reload --host 0.0.0.0 --port 8000


## API Usage

### Endpoint: Parse Resume

*POST* /api/parse-resume

*Request Body:*
json
{
  "resume_text": "Rohit Shakya. Senior Python Developer 2024-2025"
}


*Response:*
json
{
  "candidate_name": "Rohit Shakya",
  "tech_stack": [
    "Python",
    "FastAPI",
    "TensorFlow",
    "Docker",
    "AWS"
  ],
  "years_of_experience": 1,
  "fit_score": 4,
  "fit_explanation": "Excellent match for Senior Python Developer role...",
  "matching_skills": [
    "Python",
    "FastAPI",
    "Docker"
  ],
  "missing_skills": [
    "PostgreSQL",
    "Redis"
  ]
}



## Job Requirements Logic

The API evaluates candidates against hardcoded Senior Python Developer requirements:

*Must-Have Skills (50% weight):*
- Python, FastAPI, Flask, Django
- REST API, SQL, Git

*Preferred Skills (30% weight):*
- AI/ML: TensorFlow, PyTorch, LangChain
- Cloud: AWS, Azure, GCP
- Infrastructure: Docker, Kubernetes
- Databases: Redis, PostgreSQL, MongoDB
- Testing: pytest, CI/CD

*Experience Requirement (20% weight):*
- Minimum: 5 years

*Scoring Formula:*

Final Score = (Must-Have Match × 5) + (Preferred Match × 3) + (Experience Factor × 2)



## Testing

### Via Swagger UI
1. Navigate to http://localhost:8000/docs
2. Click on POST /api/parse-resume
3. Click "Try it out"
4. Paste resume text
5. Click "Execute"

### Via Web UI
1. Navigate to http://localhost:5500
2. load sample resume
3. Click "Analyze"
4. View structured results


# Run with Docker
## 1. Pull the image
docker pull rohitkumarshakya/res

## 2. Run the container (replace with your actual Groq API key)
docker run -d \
  -p 8080:8000 \
  -e GROQ_API_KEY=your_groq_api_key_here \
  rohitkumarshakya/res

# Access the API

OpenAPI Docs → http://localhost:8080/docs
Health Check → http://localhost:8080/health