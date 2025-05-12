"""Configuration settings for the medical chatbot."""
import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# LLM provider settings
LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "ollama"),  # Default to local Ollama if no env var set
    "models": {
        "claude": {
            "model_name": os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229"),
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        },
        "gemma": {
            "model_name": os.getenv("GEMMA_MODEL", "gemini-pro"),
            "api_key": os.getenv("GOOGLE_API_KEY"),
        },
        "ollama": {
            "model_name": os.getenv("OLLAMA_MODEL", "mistral"),
            "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        },
        "gpt2": {  # Keep GPT-2 as a fallback option
            "model_name": "gpt2",
            "max_length": 100,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
        }
    },
    "system_prompts": {
        "medical": """You are an AI medical assistant helping users understand their symptoms. 
        Provide helpful, accurate medical information based on the user's symptoms. 
        If the symptoms suggest a serious condition, recommend seeking medical attention.
        Always provide clear, factual information and avoid speculation.""",
        "general": """You are a helpful AI assistant answering general medical questions.
        Provide informative, accurate responses based on well-established medical knowledge.
        If you don't know something, admit it rather than providing potentially incorrect information.""",
        "emergency": """You are an emergency medical assistant. 
        Your primary goal is to help the user get appropriate emergency care.
        If symptoms suggest a life-threatening condition, urgently recommend calling emergency services."""
    },
    "cache_enabled": True,
    "cache_max_size": 1000,
}

# Model settings (legacy GPT-2 support)
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
    "window_size": (1000, 700),
    "font_size": 12,
    "theme": "light",
    "primary_color": "#4A90E2",  # Medical blue
    "secondary_color": "#E2F0FF",  # Light blue
    "accent_color": "#FF5252",  # Alert red
    "chat_background": "#FFFFFF",  # White
    "user_message_color": "#E2F0FF",  # Light blue
    "bot_message_color": "#F0F0F0",  # Light gray
    "font_family": "Arial",
    "logo_path": str(BASE_DIR / "assets" / "logo.png"),
    "enable_voice_input": False,  # Default to disabled until implemented
    "enable_voice_output": False,  # Default to disabled until implemented
}

# Authentication settings
AUTH_CONFIG = {
    "enabled": False,  # Default to disabled until implemented
    "require_login": False,
    "session_timeout": 3600,  # 1 hour
    "database_collection": "users",
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": str(BASE_DIR / "logs" / "medical_chatbot.log"),
    "max_size": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5,
} 