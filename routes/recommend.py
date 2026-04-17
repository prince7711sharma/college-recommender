"""
Recommendation route — POST /recommend-college endpoint.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from models.schemas import StudentProfile
from agents.workflow_agent import WorkflowAgent

router = APIRouter()
workflow_agent = WorkflowAgent()


@router.post("/recommend-college")
async def recommend_college(student: StudentProfile):
    """
    Recommend top colleges based on student profile.
    Uses 4-agent architecture: API → Error → UI → Workflow.
    Returns 503 with structured JSON when the LLM is unavailable.
    """
    student_data = student.model_dump()
    result = await workflow_agent.run(student_data)

    # If workflow returned an error dict, respond with correct HTTP status
    if isinstance(result, dict) and result.get("error"):
        http_status = result.get("status_code", 503)
        return JSONResponse(status_code=http_status, content=result)

    return result


@router.get("/metadata")
async def get_metadata():
    """Return available locations, courses, branches for the UI dropdowns."""
    return workflow_agent.api_agent.get_metadata()
