"""Configuration settings for the medical chatbot."""
import os
from pathlib import Path
from typing import Dict, List

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Model settings
MODEL_CONFIG = {
    "model_name": "gpt2",
    "max_length": 100,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
}

# Database settings
DB_CONFIG = {
    "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
    "db_name": "clinic_chatbot",
    "collection_name": "conversations",
}

# Medical keywords and their associated questions
MEDICAL_KEYWORDS: List[str] = [
    "headache",
    "fever",
    "pain",
    "cough",
    "diabetes",
    "asthma",
    "flu",
    "allergy",
    "fatigue",
    "dizziness",
    "nausea",
    "hypertension",
    "cholesterol",
    "cancer",
    "tumor",
    "infection",
    "virus",
    "seizure",
    "arthritis",
    "depression",
    "anxiety",
    "migraine",
    "stroke",
    "heart attack",
]

SYMPTOM_QUESTIONS: Dict[str, List[str]] = {
    "headache": [
        "Can you describe the severity of your headache?",
        "When did it start?",
        "Is it continuous or intermittent?",
    ],
    "fever": [
        "What was your last body temperature?",
        "When did the fever start?",
        "Do you have chills or sweats?",
    ],
    "diabetes": [
        "How often do you monitor blood sugar?",
        "Do you have excessive thirst or urination?",
    ],
    "asthma": [
        "How often do you experience shortness of breath?",
        "Are symptoms triggered by exercise, allergens, etc.?",
    ],
    "flu": [
        "When did your symptoms start?",
        "Do you have respiratory symptoms like cough, sore throat?",
    ],
    "allergy": [
        "What symptoms do you experience?",
        "Are your symptoms seasonal or year-round?",
    ],
}

MEDICAL_RECOMMENDATIONS: Dict[str, str] = {
    "headache": "Rest, stay hydrated, and avoid bright lights. If it persists, consult a doctor.",
    "fever": "Monitor your temperature and stay hydrated. If high or persistent, seek medical help.",
    "default": "Consult with a doctor for further assessment.",
}

# GUI settings
GUI_CONFIG = {
    "window_title": "Medical Chatbot",
    "window_size": (800, 600),
    "font_size": 12,
    "theme": "light",
} 