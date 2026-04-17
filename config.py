"""
Application configuration — loads environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "colleges.json")
