"""Main entry point for the medical chatbot application."""
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from chatbot.core import LocalChatbot, ChatbotConfig
from gui.main_window import ChatbotGUI
from config.settings import MODEL_CONFIG, DB_CONFIG

def main() -> None:
    """Main entry point for the application."""
    # Create application
    app = QApplication(sys.argv)

    # Initialize chatbot
    config = ChatbotConfig(
        model_name=MODEL_CONFIG["model_name"],
        mongo_uri=DB_CONFIG["uri"],
        db_name=DB_CONFIG["db_name"],
        collection_name=DB_CONFIG["collection_name"],
    )
    chatbot = LocalChatbot(config)

    # Create and show GUI
    gui = ChatbotGUI(chatbot)
    gui.show()

    # Start application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 