"""
Error Agent — centralized error handling and validation.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Any, Dict


class ErrorAgent:
    """Agent responsible for handling all errors gracefully."""

    @staticmethod
    def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle Pydantic validation errors with clear messages."""
        errors = []
        for error in exc.errors():
            field = " → ".join(str(loc) for loc in error["loc"])
            errors.append({
                "field": field,
                "message": error["msg"],
                "type": error["type"]
            })
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation Error",
                "detail": "Your input has issues. Please fix the following:",
                "validation_errors": errors,
                "status_code": 422
            }
        )

    @staticmethod
    def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle any unhandled exceptions."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": "Something went wrong. Please try again later.",
                "status_code": 500
            }
        )

    @staticmethod
    def handle_empty_results(student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a friendly message when no colleges match."""
        return {
            "recommendations": [],
            "student_summary": (
                f"We couldn't find colleges matching your criteria: "
                f"{student_data.get('course', 'N/A')} in {student_data.get('location', 'N/A')} "
                f"within ₹{student_data.get('budget', 0):,}/year budget. "
                f"Try broadening your search — increase budget or try a nearby city."
            ),
            "total_matched": 0
        }

    @staticmethod
    def handle_llm_failure(error: Any) -> Dict[str, Any]:
        """Return graceful fallback when LLM fails.

        Accepts either a plain string or a dict returned by LLMService.
        Preserves the original detail message so the frontend gets useful info.
        """
        if isinstance(error, dict):
            detail_msg = error.get("detail", "AI service is currently unavailable.")
        else:
            detail_msg = str(error)

        return {
            "error": "AI Service Unavailable",
            "detail": f"Our recommendation engine is temporarily down: {detail_msg}",
            "recommendations": [],
            "status_code": 503,
        }
