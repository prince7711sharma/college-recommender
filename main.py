"""
College Recommendation System — FastAPI API Server
Runs separately from the frontend UI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from routes.recommend import router as recommend_router
from agents.error_agent import ErrorAgent

app = FastAPI(
    title="🎓 AI College Recommendation System",
    description="Intelligent college recommendations powered by Groq LLM",
    version="1.0.0"
)

# CORS — allow the frontend (running on a different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handlers
error_agent = ErrorAgent()
app.add_exception_handler(RequestValidationError, error_agent.validation_error_handler)
app.add_exception_handler(Exception, error_agent.generic_error_handler)

# Routes
app.include_router(recommend_router, prefix="/api", tags=["Recommendations"])


@app.get("/", include_in_schema=False)
async def root():
    """API root — returns service info."""
    return {"message": "College Recommendation API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "College Recommendation System"}
