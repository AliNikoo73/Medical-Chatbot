"""Main window GUI module for the medical chatbot."""
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt

from src.chatbot.core import LocalChatbot

class ChatbotGUI(QMainWindow):
    """Main window GUI for the medical chatbot."""

    def __init__(self, chatbot: LocalChatbot) -> None:
        """Initialize the GUI with a chatbot instance.

        Args:
            chatbot: The chatbot instance to use.
        """
        super().__init__()
        self.chatbot = chatbot
        self.initUI()

    def initUI(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Medical Chatbot")
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()

        # Chat log
        self.chat_log = QTextEdit(self)
        self.chat_log.setReadOnly(True)
        layout.addWidget(self.chat_log)

        # Buttons
        button_layout = QHBoxLayout()
        for label, handler in [
            ("I am not feeling good!!", self.handle_symptoms),
            ("I need a doctor right away", self.handle_emergency),
            ("I want to ask questions about my disease", self.handle_general_questions),
        ]:
            button = QPushButton(label, self)
            button.clicked.connect(handler)
            button_layout.addWidget(button)
        layout.addLayout(button_layout)

        # Input area
        input_layout = QHBoxLayout()
        self.user_input = QLineEdit(self)
        self.user_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.user_input)

        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)

        # Set the main widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def handle_symptoms(self) -> None:
        """Handle the symptoms button click."""
        self.chatbot.chat_mode = "symptoms"
        self.add_to_chat_log("Chatbot", "Please describe your symptoms.")

    def handle_emergency(self) -> None:
        """Handle the emergency button click."""
        self.chatbot.chat_mode = "emergency"
        self.add_to_chat_log("Chatbot", "Please call 911 or seek emergency care.")
        self.chatbot.handle_emergency()

    def handle_general_questions(self) -> None:
        """Handle the general questions button click."""
        self.chatbot.chat_mode = "general"
        self.add_to_chat_log("Chatbot", "You can ask me any questions about your disease.")

    def add_to_chat_log(self, sender: str, message: str) -> None:
        """Add a message to the chat log.

        Args:
            sender: The sender of the message.
            message: The message content.
        """
        self.chat_log.append(f"<b>{sender}:</b> {message}")

    def send_message(self) -> None:
        """Handle sending a message."""
        user_text = self.user_input.text().strip()
        if not user_text:
            return

        self.add_to_chat_log("You", user_text)
        self.user_input.clear()

        if self.chatbot.chat_mode == "symptoms":
            response = self.chatbot.handle_medical_conversation(user_text)
        else:
            response = "I'm here to help! Please select a mode of conversation."
        self.add_to_chat_log("Chatbot", response) 