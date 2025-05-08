"""Core chatbot module implementing the medical chatbot functionality."""
from typing import Dict, List, Optional, Tuple
import re
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import torch
from pymongo import MongoClient
from pymongo.collection import Collection
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

@dataclass
class ChatbotConfig:
    """Configuration for the chatbot."""
    model_name: str = "gpt2"
    mongo_uri: str = "mongodb://localhost:27017/"
    db_name: str = "clinic_chatbot"
    collection_name: str = "conversations"

class SymptomContext(BaseModel):
    """Context for symptom-based conversation."""
    symptoms: List[str] = []
    responses: Dict[str, List[str]] = {}
    question_index: Dict[str, int] = {}

class LocalChatbot:
    """Medical chatbot implementation with symptom analysis capabilities."""

    def __init__(self, config: Optional[ChatbotConfig] = None) -> None:
        """Initialize the chatbot with configuration.

        Args:
            config: Optional configuration for the chatbot.
        """
        self.config = config or ChatbotConfig()
        self._initialize_model()
        self._initialize_database()
        self.context = SymptomContext()
        self.chat_mode: Optional[str] = None

    def _initialize_model(self) -> None:
        """Initialize the language model and tokenizer."""
        print(f"Loading {self.config.model_name}, this may take a few minutes...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        print("Model loaded successfully!")

    def _initialize_database(self) -> None:
        """Initialize MongoDB connection and collection."""
        self.client = MongoClient(self.config.mongo_uri)
        self.collection: Collection = self.client[self.config.db_name][self.config.collection_name]

    def extract_intents_and_entities(self, text: str) -> List[str]:
        """Extract medical keywords from the input text.

        Args:
            text: Input text to analyze.

        Returns:
            List of medical keywords found in the text.
        """
        medical_keywords = [
            'headache', 'fever', 'pain', 'cough', 'diabetes', 'asthma', 'flu', 'allergy',
            'fatigue', 'dizziness', 'nausea', 'hypertension', 'cholesterol', 'cancer',
            'tumor', 'infection', 'virus', 'seizure', 'arthritis', 'depression', 'anxiety',
            'migraine', 'stroke', 'heart attack'
        ]
        return [word for word in medical_keywords if re.search(rf'\b{word}\b', text, re.IGNORECASE)]

    def handle_medical_conversation(self, user_text: str) -> str:
        """Handle medical conversation based on user input.

        Args:
            user_text: User's input text.

        Returns:
            Bot's response to the user.
        """
        if entities := self.extract_intents_and_entities(user_text):
            new_symptom = entities[0]
            if new_symptom not in self.context.symptoms:
                self.context.symptoms.append(new_symptom)
                self.context.responses[new_symptom] = []
                self.context.question_index[new_symptom] = 0
                self.save_conversation(new_symptom, "start", "New symptom conversation started.")
            return self.ask_question(new_symptom)

        for symptom in self.context.symptoms:
            if self.context.question_index[symptom] >= len(self.get_symptom_questions(symptom)):
                return self.provide_recommendation(symptom)

            self.context.responses[symptom].append(user_text)
            self.save_conversation(symptom, "user", user_text)
            self.context.question_index[symptom] += 1
            return self.ask_question(symptom)
        return self.generate_response(user_text)

    def get_symptom_questions(self, symptom: str) -> List[str]:
        """Get questions for a specific symptom.

        Args:
            symptom: The symptom to get questions for.

        Returns:
            List of questions for the symptom.
        """
        questions_bank = {
            'headache': [
                "Can you describe the severity of your headache?",
                "When did it start?",
                "Is it continuous or intermittent?"
            ],
            'fever': [
                "What was your last body temperature?",
                "When did the fever start?",
                "Do you have chills or sweats?"
            ],
            'diabetes': [
                "How often do you monitor blood sugar?",
                "Do you have excessive thirst or urination?"
            ],
            'asthma': [
                "How often do you experience shortness of breath?",
                "Are symptoms triggered by exercise, allergens, etc.?"
            ],
            'flu': [
                "When did your symptoms start?",
                "Do you have respiratory symptoms like cough, sore throat?"
            ],
            'allergy': [
                "What symptoms do you experience?",
                "Are your symptoms seasonal or year-round?"
            ]
        }
        return questions_bank.get(symptom, ["Please provide more details about your symptom."])

    def provide_recommendation(self, symptom: str) -> str:
        """Provide medical recommendation for a symptom.

        Args:
            symptom: The symptom to provide recommendation for.

        Returns:
            Medical recommendation for the symptom.
        """
        recommendations = {
            'headache': "Rest, stay hydrated, and avoid bright lights. If it persists, consult a doctor.",
            'fever': "Monitor your temperature and stay hydrated. If high or persistent, seek medical help."
        }
        recommendation = recommendations.get(symptom, "Consult with a doctor for further assessment.")
        self.save_conversation(symptom, "bot", recommendation)
        return recommendation

    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate a response using the language model.

        Args:
            prompt: Input prompt for the model.
            max_length: Maximum length of the generated response.

        Returns:
            Generated response text.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def save_conversation(self, symptom: str, role: str, message: str) -> None:
        """Save conversation to the database.

        Args:
            symptom: The symptom being discussed.
            role: The role of the message sender (user/bot).
            message: The message content.
        """
        self.collection.insert_one({
            'symptom': symptom,
            'role': role,
            'message': message
        })

    def handle_emergency(self) -> None:
        """Handle emergency situations by providing guidance and opening maps."""
        print("Emergency detected. Please call 911 or seek urgent medical care.")
        webbrowser.open("https://www.google.com/maps/search/clinic+near+me/")

    def ask_question(self, symptom: str) -> str:
        """Ask a question about a specific symptom.

        Args:
            symptom: The symptom to ask about.

        Returns:
            The question to ask.
        """
        questions = self.get_symptom_questions(symptom)
        index = self.context.question_index[symptom]
        if index < len(questions):
            return questions[index]
        return self.provide_recommendation(symptom) 