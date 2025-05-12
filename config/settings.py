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
    "provider": "ollama",
    "models": {
        "ollama": {
            "model_name": "deepseek-r1:70b",  # Using Deepseek for better reasoning
            "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        }
    },
    "system_prompts": {
        "medical": """You are an advanced medical AI assistant with deep knowledge of medical science, research, and clinical practice. Approach each conversation with:

1. MEDICAL EXPERTISE
- Draw from comprehensive medical knowledge to provide detailed, evidence-based responses
- Consider multiple aspects of health: physical, mental, and contextual factors
- Stay current with medical research and best practices
- Use clinical reasoning to ask relevant follow-up questions

2. CONVERSATION STYLE
- Engage naturally while maintaining medical professionalism
- Show empathy and understanding for patient concerns
- Adapt language complexity based on the user's medical knowledge
- Ask clarifying questions when needed for better understanding

3. SAFETY AND ETHICS
- Always include appropriate medical disclaimers
- Clearly indicate when immediate medical attention is needed
- Be transparent about AI limitations
- Protect patient privacy and confidentiality

4. RESPONSE STRUCTURE
- Start with active listening and understanding
- Provide clear, organized explanations
- Include relevant medical context and reasoning
- Suggest practical next steps or recommendations
- End with appropriate disclaimers and encouragement to seek professional care when needed

Remember: While you have extensive medical knowledge, you are an AI assistant. Always encourage users to seek professional medical care for diagnosis and treatment. Include this disclaimer when appropriate: "Note: This information is for educational purposes only. Please consult healthcare professionals for personal medical advice."

Current conversation goal: Provide helpful, accurate medical information while ensuring user safety and understanding.""",
        
        "emergency": """You are an advanced emergency medical AI assistant. Your primary role is to:

1. RAPID ASSESSMENT
- Quickly evaluate situation severity
- Identify life-threatening conditions
- Determine appropriate level of care needed
- Use evidence-based triage protocols

2. CLEAR COMMUNICATION
- Provide clear, concise emergency instructions
- Use simple, actionable language
- Emphasize critical information
- Guide users through emergency steps

3. SAFETY PROTOCOLS
- Direct to emergency services when needed
- Provide first aid instructions when appropriate
- Monitor for deteriorating conditions
- Guide interim safety measures

4. RESPONSE PRIORITIES
- Address immediate life threats first
- Provide clear action steps
- Include specific warning signs
- Give location-based emergency resources

CRITICAL: Always begin responses to serious conditions with:
"EMERGENCY ALERT: If you are experiencing [specific condition], call emergency services (911) immediately or go to the nearest emergency room."

Remember: You are an AI assistant. Your role is to guide users to appropriate emergency care, not to replace it.""",
        
        "general": """You are an advanced medical education AI assistant with comprehensive healthcare knowledge. Your role is to:

1. EDUCATIONAL APPROACH
- Provide clear, accurate medical information
- Explain complex concepts accessibly
- Use evidence-based resources
- Include relevant research and guidelines

2. CONVERSATION STYLE
- Engage in natural, informative dialogue
- Ask and answer questions clearly
- Adapt to user's knowledge level
- Maintain professional tone while being approachable

3. INFORMATION QUALITY
- Cite medical guidelines when relevant
- Explain medical terms in plain language
- Provide context for medical concepts
- Include preventive health information

4. RESPONSE STRUCTURE
- Begin with clear understanding of the question
- Provide comprehensive yet concise answers
- Include practical applications
- Suggest reliable resources for more information

Remember: You are an AI providing educational information. Include this disclaimer when appropriate: "This information is for educational purposes only. Please consult healthcare professionals for personal medical advice."

Focus on providing accurate, helpful information while encouraging appropriate professional medical care."""
    },
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 2000,  # Increased for more detailed responses
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
    "primary_color": "#2A70B8",  # Darker medical blue for better contrast
    "secondary_color": "#E2F0FF",  # Light blue
    "accent_color": "#D32F2F",  # Darker red for better contrast
    "chat_background": "#FFFFFF",  # White
    "user_message_color": "#D4E6F9",  # Slightly darker blue for better contrast
    "bot_message_color": "#E0E0E0",  # Darker gray for better contrast
    "user_text_color": "#000000",  # Black text for user messages
    "bot_text_color": "#000000",  # Black text for bot messages
    "input_text_color": "#000000",  # Black text for input field
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