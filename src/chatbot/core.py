"""Core chatbot module implementing the medical chatbot functionality."""
from typing import Dict, List, Optional, Tuple, Any
import re
import os
import webbrowser
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from pymongo import MongoClient
from pymongo.collection import Collection
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

from src.chatbot.llm_provider import LLMProvider, LLMFactory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatbotConfig:
    """Configuration for the chatbot."""
    # LLM settings
    llm_provider: str = "gpt2"
    llm_model_name: str = "gpt2"
    llm_api_key: Optional[str] = None
    llm_host: Optional[str] = None
    
    # MongoDB settings
    mongo_uri: str = "mongodb://localhost:27017/"
    db_name: str = "clinic_chatbot"
    collection_name: str = "conversations"
    
    # Cache settings
    cache_enabled: bool = True
    
    # System prompts
    system_prompts: Dict[str, str] = None

class SymptomContext(BaseModel):
    """Context for symptom-based conversation."""
    symptoms: List[str] = []
    responses: Dict[str, List[str]] = {}
    question_index: Dict[str, int] = {}
    conversation_history: List[Dict[str, str]] = []

class LocalChatbot:
    """Medical chatbot implementation with symptom analysis capabilities."""

    def __init__(self, config: Optional[ChatbotConfig] = None) -> None:
        """Initialize the chatbot with configuration.

        Args:
            config: Optional configuration for the chatbot.
        """
        self.config = config or ChatbotConfig()
        self._initialize_llm()
        self._initialize_database()
        self.context = SymptomContext()
        self.chat_mode: Optional[str] = None

    def _initialize_llm(self) -> None:
        """Initialize the language model based on provider configuration."""
        try:
            if self.config.llm_provider == "gpt2":
                # Legacy GPT-2 initialization
                logger.info(f"Loading {self.config.llm_model_name}, this may take a few minutes...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_name)
                self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
                self.llm_provider = None  # No provider for legacy model
                logger.info("Model loaded successfully!")
            else:
                # Initialize LLM provider from factory
                logger.info(f"Initializing {self.config.llm_provider} provider...")
                provider_kwargs = {
                    "model": self.config.llm_model_name
                }
                
                # Add API key if present
                if self.config.llm_api_key:
                    provider_kwargs["api_key"] = self.config.llm_api_key
                
                # Add host for Ollama
                if self.config.llm_provider == "ollama" and self.config.llm_host:
                    provider_kwargs["host"] = self.config.llm_host
                
                self.llm_provider = LLMFactory.create_provider(
                    self.config.llm_provider, 
                    **provider_kwargs
                )
                logger.info(f"Using {self.llm_provider.get_name()} for responses")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            # Fallback to GPT-2 if provider initialization fails
            logger.info("Falling back to GPT-2...")
            self.config.llm_provider = "gpt2"
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
            self.llm_provider = None

    def _initialize_database(self) -> None:
        """Initialize MongoDB connection and collection."""
        try:
            self.client = MongoClient(self.config.mongo_uri)
            self.collection: Collection = self.client[self.config.db_name][self.config.collection_name]
            logger.info(f"Connected to MongoDB: {self.config.db_name}.{self.config.collection_name}")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            self.client = None
            self.collection = None

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
        # Add user message to conversation history
        self.context.conversation_history.append({"role": "user", "content": user_text})
        
        # Extract symptoms if any
        if entities := self.extract_intents_and_entities(user_text):
            new_symptom = entities[0]
            if new_symptom not in self.context.symptoms:
                self.context.symptoms.append(new_symptom)
                self.context.responses[new_symptom] = []
                self.context.question_index[new_symptom] = 0
                self.save_conversation(new_symptom, "start", "New symptom conversation started.")
            response = self.ask_question(new_symptom)
        else:
            # Check for existing symptom conversation
            for symptom in self.context.symptoms:
                if self.context.question_index[symptom] >= len(self.get_symptom_questions(symptom)):
                    response = self.provide_recommendation(symptom)
                    break
                else:
                    self.context.responses[symptom].append(user_text)
                    self.save_conversation(symptom, "user", user_text)
                    self.context.question_index[symptom] += 1
                    response = self.ask_question(symptom)
                    break
            else:
                # No symptom found, use LLM for general response
                response = self.generate_response(user_text)
        
        # Add bot response to conversation history
        self.context.conversation_history.append({"role": "assistant", "content": response})
        
        return response

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
        # If we have an LLM provider, use it for more detailed recommendations
        if self.llm_provider and symptom:
            # Build prompt based on conversation history
            symptom_history = []
            for item in self.context.conversation_history:
                if item["role"] == "user" and any(s in item["content"].lower() for s in [symptom, "symptoms", "feeling"]):
                    symptom_history.append(f"Patient: {item['content']}")
                elif item["role"] == "assistant" and "?" in item["content"]:
                    symptom_history.append(f"Doctor: {item['content']}")
                    if response := next((r for r in self.context.conversation_history if r["role"] == "user" and 
                                      self.context.conversation_history.index(r) > 
                                      self.context.conversation_history.index(item)), None):
                        symptom_history.append(f"Patient: {response['content']}")
            
            symptom_description = "\n".join(symptom_history)
            
            prompt = f"""Based on the following patient symptoms for {symptom}:
            
{symptom_description}

Provide a brief, helpful medical recommendation. Mention when they should see a doctor 
and what home care measures might help. Keep it concise and informative.
            """
            
            recommendation = self.llm_provider.generate_response(prompt)
            self.save_conversation(symptom, "bot", recommendation)
            return recommendation
            
        # Fallback to predefined recommendations
        recommendations = {
            'headache': "Rest, stay hydrated, and avoid bright lights. If it persists, consult a doctor.",
            'fever': "Monitor your temperature and stay hydrated. If high or persistent, seek medical help."
        }
        recommendation = recommendations.get(symptom, "Consult with a doctor for further assessment.")
        self.save_conversation(symptom, "bot", recommendation)
        return recommendation

    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using the configured language model.
        
        Args:
            prompt: Input prompt for the model.
            **kwargs: Additional parameters.
            
        Returns:
            Generated response text.
        """
        try:
            # Use LLM provider if available
            if self.llm_provider:
                # Format prompt with context
                if self.chat_mode and self.config.system_prompts and self.chat_mode in self.config.system_prompts:
                    system_prompt = self.config.system_prompts[self.chat_mode]
                    context_prompt = f"{system_prompt}\n\nUser question: {prompt}"
                else:
                    context_prompt = f"Answer this medical question in a helpful way: {prompt}"
                
                return self.llm_provider.generate_response(context_prompt, **kwargs)
            
            # Fallback to legacy GPT-2 model
            max_length = kwargs.get("max_length", 100)
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                top_k=kwargs.get("top_k", 50),
                top_p=kwargs.get("top_p", 0.95),
                temperature=kwargs.get("temperature", 0.7),
                pad_token_id=self.tokenizer.pad_token_id
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error generating a response. Please try again."

    def save_conversation(self, symptom: str, role: str, message: str) -> None:
        """Save conversation to the database.

        Args:
            symptom: The symptom being discussed.
            role: The role of the message sender (user/bot).
            message: The message content.
        """
        try:
            if self.collection:
                self.collection.insert_one({
                    'symptom': symptom,
                    'role': role,
                    'message': message,
                    'timestamp': self.client.server_time()
                })
        except Exception as e:
            logger.error(f"Error saving to database: {e}")

    def handle_emergency(self) -> None:
        """Handle emergency situations by providing guidance and opening maps."""
        print("Emergency detected. Please call 911 or seek urgent medical care.")
        
        try:
            # If we have an LLM provider, get emergency advice
            if self.llm_provider and self.config.system_prompts and "emergency" in self.config.system_prompts:
                prompt = f"{self.config.system_prompts['emergency']}\n\nProvide a brief emergency guidance message."
                emergency_message = self.llm_provider.generate_response(prompt)
                logger.info(f"Emergency guidance: {emergency_message}")
                print(emergency_message)
        except Exception as e:
            logger.error(f"Error generating emergency guidance: {e}")
        
        # Open map to nearby clinics
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
            question = questions[index]
            self.save_conversation(symptom, "bot", question)
            return question
        return self.provide_recommendation(symptom) 