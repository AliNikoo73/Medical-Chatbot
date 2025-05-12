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
                
                try:
                    self.llm_provider = LLMFactory.create_provider(
                        self.config.llm_provider, 
                        **provider_kwargs
                    )
                    logger.info(f"Using {self.llm_provider.get_name()} for responses")
                except ValueError as e:
                    # If API key is missing and provider is Claude or Gemma, try Ollama as fallback
                    if "API key is required" in str(e) and self.config.llm_provider in ["claude", "gemma"]:
                        logger.warning(f"Missing API key for {self.config.llm_provider}, trying Ollama instead")
                        self.config.llm_provider = "ollama"
                        self.config.llm_model_name = "mistral"
                        provider_kwargs = {
                            "model": self.config.llm_model_name,
                            "host": self.config.llm_host or "http://localhost:11434"
                        }
                        self.llm_provider = LLMFactory.create_provider(
                            self.config.llm_provider, 
                            **provider_kwargs
                        )
                        logger.info(f"Using {self.llm_provider.get_name()} for responses")
                    else:
                        # Re-raise if it's not an API key issue or Ollama is not available
                        raise
                        
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
        """Extract medical keywords and intents from the input text.

        Args:
            text: Input text to analyze.

        Returns:
            List of medical keywords found in the text.
        """
        # Enhanced list of medical keywords and conditions
        medical_keywords = [
            # Symptoms
            'headache', 'fever', 'pain', 'cough', 'fatigue', 'dizziness', 'nausea',
            'vomiting', 'diarrhea', 'constipation', 'rash', 'swelling', 'bleeding',
            'bruising', 'numbness', 'tingling', 'weakness', 'stiffness', 'tremor',
            
            # Conditions
            'diabetes', 'asthma', 'hypertension', 'arthritis', 'depression', 'anxiety',
            'migraine', 'insomnia', 'allergies', 'infection', 'inflammation',
            'cancer', 'tumor', 'stroke', 'heart attack', 'seizure',
            
            # Body parts
            'head', 'chest', 'stomach', 'back', 'neck', 'throat', 'joints',
            'muscles', 'skin', 'eyes', 'ears', 'nose', 'mouth', 'teeth',
            
            # Qualifiers
            'severe', 'chronic', 'acute', 'persistent', 'recurring', 'intermittent',
            'mild', 'moderate', 'intense', 'sharp', 'dull', 'throbbing'
        ]
        
        # Find all matches (case-insensitive)
        matches = []
        text_lower = text.lower()
        for keyword in medical_keywords:
            if re.search(rf'\b{keyword}\b', text_lower):
                matches.append(keyword)
        
        return matches

    def handle_medical_conversation(self, user_text: str) -> str:
        """Handle medical conversation based on user input.

        Args:
            user_text: User's input text.

        Returns:
            Bot's response to the user.
        """
        # Add user message to conversation history
        self.context.conversation_history.append({"role": "user", "content": user_text})
        
        # Extract medical entities and update context
        if entities := self.extract_intents_and_entities(user_text):
            for entity in entities:
                if entity not in self.context.symptoms:
                    self.context.symptoms.append(entity)
                    self.context.responses[entity] = []
                self.context.responses[entity].append(user_text)
                self.save_conversation(entity, "user", user_text)
        
        # Generate response using the enhanced context
        response = self.generate_response(user_text)
        
        # Save the conversation
        for symptom in self.context.symptoms:
            self.save_conversation(symptom, "bot", response)
        
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
            # Format prompt with context and conversation history
            context_prompt = self._build_context_prompt(prompt)
            
            # Generate response using Ollama
            if self.llm_provider:
                response = self.llm_provider.generate_response(context_prompt, **kwargs)
                
                # Post-process response
                response = self._post_process_response(response)
                
                # Save to conversation history
                self.context.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                return response
            else:
                return "I apologize, but I'm currently unable to process requests. Please try again later."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."

    def _build_context_prompt(self, user_input: str) -> str:
        """Build a context-aware prompt for the model.
        
        Args:
            user_input: The user's input text.
            
        Returns:
            A formatted prompt string.
        """
        # Get appropriate system prompt based on chat mode
        if self.chat_mode and self.config.system_prompts:
            system_prompt = self.config.system_prompts.get(self.chat_mode, "")
        else:
            system_prompt = self.config.system_prompts.get("general", "")
        
        # Build conversation history
        conversation = ""
        if self.context.conversation_history:
            # Include up to last 10 exchanges for context
            recent_history = self.context.conversation_history[-10:]
            for msg in recent_history:
                role = "Patient" if msg["role"] == "user" else "Assistant"
                conversation += f"{role}: {msg['content']}\n\n"  # Added extra newline for clarity
        
        # Add current symptoms context if available
        symptoms_context = ""
        if self.context.symptoms:
            symptoms_context = "\nCurrent Health Context:\n"
            for symptom in self.context.symptoms:
                responses = self.context.responses.get(symptom, [])
                if responses:
                    # Format the symptom information more naturally
                    symptoms_context += f"- Patient reported {symptom}: {' | '.join(responses)}\n"
        
        # Add any relevant medical keywords found in the current message
        current_keywords = self.extract_intents_and_entities(user_input)
        if current_keywords:
            symptoms_context += "\nNew symptoms/conditions mentioned: " + ", ".join(current_keywords)
        
        # Combine all elements with improved formatting
        full_prompt = f"""{system_prompt}

CONVERSATION HISTORY:
{conversation}

{symptoms_context}

Current Patient Message: {user_input}

Instructions:
1. Analyze the patient's message and medical context
2. If clarification is needed, ask specific follow-up questions
3. Provide a detailed, evidence-based response
4. Include relevant medical information and explanations
5. Suggest appropriate next steps or recommendations
6. Add appropriate medical disclaimers

Respond in a natural, conversational manner while maintaining medical professionalism."""

        return full_prompt

    def _post_process_response(self, response: str) -> str:
        """Post-process the model's response.
        
        Args:
            response: The raw response from the model.
            
        Returns:
            Processed response string.
        """
        # Ensure response has appropriate disclaimer
        if self.chat_mode == "emergency":
            if "EMERGENCY WARNING:" not in response:
                response += "\n\nEMERGENCY WARNING: If you are experiencing a medical emergency, immediately call your local emergency services (911 in the US) or go to the nearest emergency room. This AI system is not a substitute for emergency medical care."
        elif "Note: This information is for educational purposes only" not in response:
            response += "\n\nNote: This information is for educational purposes only and not a substitute for professional medical advice. Please consult a healthcare provider for diagnosis and treatment."
        
        return response

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