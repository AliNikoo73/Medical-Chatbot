"""Main entry point for the medical chatbot application."""
import sys
import os
import logging
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from src.chatbot.core import LocalChatbot, ChatbotConfig
from src.gui.main_window import ChatbotGUI
from config.settings import MODEL_CONFIG, DB_CONFIG, LLM_CONFIG, GUI_CONFIG, LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG.get("level", "INFO")),
    format=LOGGING_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Main entry point for the application."""
    logger.info("Starting Medical Chatbot application")
    
    # Create application
    app = QApplication(sys.argv)
    
    # Use environment variables if available, otherwise use settings
    llm_provider = os.getenv("LLM_PROVIDER", LLM_CONFIG.get("provider", "gpt2"))
    
    # Get model configurations based on provider
    model_settings = LLM_CONFIG.get("models", {}).get(llm_provider, {})
    model_name = model_settings.get("model_name", MODEL_CONFIG.get("model_name", "gpt2"))
    
    # Log LLM provider info
    logger.info(f"Using LLM provider: {llm_provider}, model: {model_name}")
    
    # Initialize chatbot with configuration
    config = ChatbotConfig(
        llm_provider=llm_provider,
        llm_model_name=model_name,
        llm_api_key=model_settings.get("api_key"),
        llm_host=model_settings.get("host") if llm_provider == "ollama" else None,
        mongo_uri=DB_CONFIG["uri"],
        db_name=DB_CONFIG["db_name"],
        collection_name=DB_CONFIG["collection_name"],
        system_prompts=LLM_CONFIG.get("system_prompts"),
    )
    
    try:
        chatbot = LocalChatbot(config)
        
        # Create and show GUI
        logger.info("Initializing GUI")
        gui = ChatbotGUI(chatbot)
        gui.show()
        
        # Start application
        logger.info("Application started")
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 