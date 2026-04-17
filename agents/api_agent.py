"""
API Agent — handles the /recommend-college endpoint logic.
"""

from typing import Dict, Any
from services.college_service import CollegeService
from services.llm_service import LLMService


class APIAgent:
    """Agent responsible for orchestrating the recommendation API flow."""

    def __init__(self):
        self.college_service = CollegeService()
        self.llm_service = LLMService()

    async def process_recommendation(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full pipeline:
        1. Filter colleges from dataset
        2. Pass to LLM for ranking
        3. Return structured response
        """
        # Step 1: Filter
        filtered = self.college_service.filter_colleges(
            location=student_data["location"],
            course=student_data["course"],
            budget=student_data["budget"],
            marks=student_data["marks"],
            branch=student_data.get("branch"),
            category=student_data.get("category", "general")
        )

        if not filtered:
            return {
                "recommendations": [],
                "student_summary": f"No colleges found matching: {student_data['course']} in {student_data['location']} within ₹{student_data['budget']:,} budget.",
                "total_matched": 0
            }

        # Step 2: LLM ranking
        result = self.llm_service.get_recommendations(student_data, filtered)

        # Step 3: Attach metadata
        result["total_matched"] = len(filtered)
        return result

    def get_metadata(self) -> Dict[str, Any]:
        """Return available filter options and dataset statistics for the UI."""
        stats = self.college_service.get_stats()
        return {
            # Dropdown options
            "locations": self.college_service.get_all_locations(),
            "courses":   self.college_service.get_all_courses(),
            "branches":  self.college_service.get_all_branches(),
            "types":     self.college_service.get_all_types(),
            # Dataset statistics
            "total_colleges":       stats["total_colleges"],
            "government_colleges":  stats["government_colleges"],
            "private_colleges":     stats["private_colleges"],
            "colleges_with_hostel": stats["colleges_with_hostel"],
            "states_covered":       stats["states_covered"],
            "states":               stats["states"],
            "naac_distribution":    stats["naac_distribution"],
        }
