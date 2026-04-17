"""
Pydantic models for request/response validation — matches updated colleges.json structure.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class StudentProfile(BaseModel):
    """Input model for student recommendation request."""
    marks: int = Field(..., ge=0, le=100, description="Student marks as percentage (0-100)")
    budget: int = Field(..., gt=0, description="Maximum budget in INR per year")
    location: str = Field(..., min_length=1, description="Preferred city or state")
    course: str = Field(..., min_length=1, description="Desired course (e.g., B.Tech, MBA)")
    branch: Optional[str] = Field(None, description="Preferred branch (e.g., Computer Science)")
    category: Optional[str] = Field("general", description="Reservation category: general/obc/sc/st")


class CutoffRank(BaseModel):
    general: int
    obc: int
    sc: int
    st: int


class College(BaseModel):
    """Model representing a single college from the dataset."""
    id: int
    name: str
    state: str
    city: str
    type: str
    courses: List[str]
    branches: List[str]
    fees_per_year: int
    entrance_exam: List[str]
    cutoff_rank: CutoffRank
    placement_avg_lpa: int
    placement_highest_lpa: int
    hostel: bool
    naac_grade: str
    website: str
    contact: str
    facilities: List[str]
    description: str


class CollegeRecommendation(BaseModel):
    """Single college recommendation in the response."""
    name: str
    city: str
    state: str
    type: str
    fees_per_year: str
    branches_available: List[str]
    entrance_exam: List[str]
    placement_avg_lpa: int
    naac_grade: str
    hostel: bool
    website: str
    reason: str


class RecommendationResponse(BaseModel):
    """API response model."""
    recommendations: List[CollegeRecommendation]
    student_summary: Optional[str] = None
    total_matched: int = 0


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
    status_code: int
