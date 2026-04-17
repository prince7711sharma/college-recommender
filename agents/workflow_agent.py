"""
Workflow Agent — manages the end-to-end pipeline and logging.
"""

import time
import logging
from typing import Dict, Any
from agents.api_agent import APIAgent
from agents.error_agent import ErrorAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("WorkflowAgent")


class WorkflowAgent:
    """Agent that orchestrates the full recommendation workflow."""

    def __init__(self):
        self.api_agent = APIAgent()
        self.error_agent = ErrorAgent()

    async def run(self, student_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full recommendation workflow."""
        start_time = time.time()
        logger.info(f"🚀 Workflow started for: {student_data}")

        try:
            logger.info("📊 Step 1: Filtering colleges from dataset...")
            result = await self.api_agent.process_recommendation(student_data)

            # 1️⃣ LLM returned a hard error dict  (has "error" key)
            if result.get("error"):
                logger.error(
                    f"❌ LLM error: {result.get('error')} — {result.get('detail', '')}"
                )
                result = self.error_agent.handle_llm_failure(result)

            # 2️⃣ LLM was unavailable → static mock was used (_is_mock flag)
            elif result.get("_is_mock"):
                logger.warning("⚠️ LLM unavailable — mock response used.")
                result = self.error_agent.handle_llm_failure(
                    "The AI recommendation engine is temporarily unavailable. "
                    "Please try again in a few minutes."
                )

            # 3️⃣ Real empty result: no colleges matched the filter criteria
            elif not result.get("recommendations"):
                logger.warning("⚠️ No colleges matched filters — returning empty fallback.")
                result = self.error_agent.handle_empty_results(student_data)

            elapsed = round(time.time() - start_time, 2)
            result["processing_time_seconds"] = elapsed
            logger.info(
                f"✅ Workflow completed in {elapsed}s — "
                f"{len(result.get('recommendations', []))} recommendations"
            )
            return result

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.error(f"💥 Workflow crashed after {elapsed}s: {str(e)}")
            result = self.error_agent.handle_llm_failure(str(e))
            result["processing_time_seconds"] = elapsed
            return result
