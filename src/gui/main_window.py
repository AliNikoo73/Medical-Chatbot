"""Main window GUI module for the medical chatbot."""
import os
import sys
import logging
from typing import Optional, List, Dict
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QFrame,
    QScrollArea,
    QSplitter,
    QComboBox,
    QTabWidget,
    QStatusBar,
    QMenu,
    QAction,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QSize, QUrl, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QPixmap, QColor, QPalette, QTextCursor, QDesktopServices

from src.chatbot.core import LocalChatbot
from config.settings import GUI_CONFIG

# Setup logging
logger = logging.getLogger(__name__)

class MessageWidget(QFrame):
    """Widget for displaying a single message in the chat."""
    
    def __init__(self, sender: str, message: str, is_user: bool = False, parent=None):
        """Initialize a message widget.
        
        Args:
            sender: The sender of the message.
            message: The message content.
            is_user: Whether the message is from the user.
            parent: The parent widget.
        """
        super().__init__(parent)
        self.sender = sender
        self.message = message
        self.is_user = is_user
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface."""
        # Set frame properties
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        
        # Set background color based on sender
        palette = self.palette()
        if self.is_user:
            palette.setColor(QPalette.Window, QColor(GUI_CONFIG.get("user_message_color", "#E2F0FF")))
        else:
            palette.setColor(QPalette.Window, QColor(GUI_CONFIG.get("bot_message_color", "#F0F0F0")))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Sender label
        sender_label = QLabel(f"<b>{self.sender}</b>")
        sender_font = sender_label.font()
        sender_font.setPointSize(GUI_CONFIG.get("font_size", 12))
        sender_label.setFont(sender_font)
        layout.addWidget(sender_label)
        
        # Message label
        message_label = QLabel(self.message)
        message_label.setWordWrap(True)
        message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        message_font = message_label.font()
        message_font.setPointSize(GUI_CONFIG.get("font_size", 12) - 1)
        message_label.setFont(message_font)
        layout.addWidget(message_label)
        
        self.setLayout(layout)

class ChatLogWidget(QScrollArea):
    """Widget for displaying the chat log."""
    
    def __init__(self, parent=None):
        """Initialize the chat log widget."""
        super().__init__(parent)
        self.message_widgets = []
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface."""
        # Set scroll area properties
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create a container widget for messages
        self.container = QWidget()
        self.container.setObjectName("chatContainer")
        self.container.setStyleSheet(
            f"QWidget#chatContainer {{ background-color: {GUI_CONFIG.get('chat_background', '#FFFFFF')}; }}"
        )
        
        # Set layout for container
        self.message_layout = QVBoxLayout(self.container)
        self.message_layout.setAlignment(Qt.AlignTop)
        self.message_layout.setSpacing(10)
        self.message_layout.setContentsMargins(10, 10, 10, 10)
        
        # Set the container as the widget for the scroll area
        self.setWidget(self.container)
        
    def add_message(self, sender: str, message: str, is_user: bool = False):
        """Add a message to the chat log.
        
        Args:
            sender: The sender of the message.
            message: The message content.
            is_user: Whether the message is from the user.
        """
        # Create a message widget
        message_widget = MessageWidget(sender, message, is_user)
        
        # Add the widget to the layout
        self.message_layout.addWidget(message_widget)
        self.message_widgets.append(message_widget)
        
        # Scroll to the bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class StatusBarWidget(QStatusBar):
    """Status bar widget for displaying application status."""
    
    def __init__(self, parent=None):
        """Initialize the status bar widget."""
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """Initialize the user interface."""
        # Set status bar properties
        self.setStyleSheet(
            f"QStatusBar {{ color: #666666; font-size: {GUI_CONFIG.get('font_size', 12) - 2}pt; }}"
        )
        
        # Add model info label
        self.model_info = QLabel("Model: Not initialized")
        self.addPermanentWidget(self.model_info)
        
        # Add connection status label
        self.connection_status = QLabel("MongoDB: Not connected")
        self.addPermanentWidget(self.connection_status)
        
    def set_model_info(self, model_name: str):
        """Set the model info label text."""
        self.model_info.setText(f"Model: {model_name}")
        
    def set_connection_status(self, status: bool):
        """Set the connection status label text."""
        if status:
            self.connection_status.setText("MongoDB: Connected")
            self.connection_status.setStyleSheet("color: green;")
        else:
            self.connection_status.setText("MongoDB: Not connected")
            self.connection_status.setStyleSheet("color: red;")

class ChatbotGUI(QMainWindow):
    """Main window GUI for the medical chatbot."""

    def __init__(self, chatbot: LocalChatbot) -> None:
        """Initialize the GUI with a chatbot instance.

        Args:
            chatbot: The chatbot instance to use.
        """
        super().__init__()
        self.chatbot = chatbot
        
        # Set application font
        font_family = GUI_CONFIG.get("font_family", "Arial")
        font_size = GUI_CONFIG.get("font_size", 12)
        self.app_font = QFont(font_family, font_size)
        QApplication.setFont(self.app_font)
        
        # Initialize UI
        self.initUI()
        
        # Update status bar
        self.update_status_bar()
        
        # Add welcome message
        self.add_to_chat_log("Medical Chatbot", "Welcome to the Medical Chatbot. How can I help you today?")

    def initUI(self) -> None:
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle(GUI_CONFIG.get("window_title", "Medical Chatbot"))
        window_size = GUI_CONFIG.get("window_size", (1000, 700))
        self.setGeometry(100, 100, window_size[0], window_size[1])
        
        # Set window style
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: white;
            }}
            QPushButton {{
                background-color: {GUI_CONFIG.get("primary_color", "#4A90E2")};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #3A80D2;
            }}
            QPushButton:pressed {{
                background-color: #2A70C2;
            }}
            QLineEdit {{
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                padding: 8px;
                selection-background-color: {GUI_CONFIG.get("primary_color", "#4A90E2")};
            }}
            QTextEdit {{
                border: 1px solid #CCCCCC;
                border-radius: 4px;
                selection-background-color: {GUI_CONFIG.get("primary_color", "#4A90E2")};
            }}
        """)
        
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Header
        header_layout = QHBoxLayout()
        
        # Logo (if available)
        logo_path = GUI_CONFIG.get("logo_path")
        if logo_path and os.path.exists(logo_path):
            logo_label = QLabel()
            logo_pixmap = QPixmap(logo_path)
            logo_label.setPixmap(logo_pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            header_layout.addWidget(logo_label)
        
        # Title
        title_label = QLabel("Medical Chatbot")
        title_font = QFont(GUI_CONFIG.get("font_family", "Arial"), GUI_CONFIG.get("font_size", 12) + 4)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"color: {GUI_CONFIG.get('primary_color', '#4A90E2')};")
        header_layout.addWidget(title_label)
        
        # Provider selection
        header_layout.addStretch()
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["GPT-2", "Claude", "Gemma", "Mistral (Ollama)"])
        self.provider_combo.setToolTip("Select the AI provider to use for responses")
        # Default to current provider
        if hasattr(self.chatbot, 'llm_provider') and self.chatbot.llm_provider:
            provider_name = self.chatbot.llm_provider.get_name()
            if "claude" in provider_name.lower():
                self.provider_combo.setCurrentIndex(1)
            elif "gemma" in provider_name.lower() or "gemini" in provider_name.lower():
                self.provider_combo.setCurrentIndex(2)
            elif "mistral" in provider_name.lower() or "ollama" in provider_name.lower():
                self.provider_combo.setCurrentIndex(3)
            else:
                self.provider_combo.setCurrentIndex(0)
        else:
            self.provider_combo.setCurrentIndex(0)
        
        # Disable changing for now until we implement dynamic switching
        self.provider_combo.setEnabled(False)
        self.provider_combo.setToolTip("Provider switching will be available in a future update")
            
        header_layout.addWidget(QLabel("Provider:"))
        header_layout.addWidget(self.provider_combo)
        
        main_layout.addLayout(header_layout)
        
        # Mode buttons
        mode_layout = QHBoxLayout()
        for label, handler, color, icon in [
            ("I am not feeling good!!", self.handle_symptoms, GUI_CONFIG.get("primary_color", "#4A90E2"), "🤒"),
            ("I need a doctor right away", self.handle_emergency, GUI_CONFIG.get("accent_color", "#FF5252"), "🚑"),
            ("I want to ask questions about my disease", self.handle_general_questions, "#4CAF50", "❓"),
        ]:
            button = QPushButton(f"{icon} {label}")
            button.clicked.connect(handler)
            button.setStyleSheet(f"background-color: {color};")
            mode_layout.addWidget(button)
        main_layout.addLayout(mode_layout)
        
        # Chat area
        self.chat_log = ChatLogWidget()
        main_layout.addWidget(self.chat_log, 1)  # 1 = stretch factor
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        self.user_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.user_input, 1)  # 1 = stretch factor
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setIcon(QIcon.fromTheme("arrow-right"))
        input_layout.addWidget(self.send_button)
        
        main_layout.addLayout(input_layout)
        
        # Status bar
        self.status_bar = StatusBarWidget()
        self.setStatusBar(self.status_bar)
        
        # Set layouts
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Create menu bar
        self.create_menu_bar()

    def create_menu_bar(self):
        """Create the application menu bar."""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        export_action = QAction("Export Conversation", self)
        export_action.setStatusTip("Export the current conversation to a file")
        export_action.triggered.connect(self.export_conversation)
        file_menu.addAction(export_action)
        
        clear_action = QAction("Clear Conversation", self)
        clear_action.setStatusTip("Clear the current conversation")
        clear_action.triggered.connect(self.clear_conversation)
        file_menu.addAction(clear_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.setStatusTip("Show the application's About box")
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def update_status_bar(self):
        """Update the status bar with current information."""
        # Update model info
        if hasattr(self.chatbot, 'llm_provider') and self.chatbot.llm_provider:
            self.status_bar.set_model_info(self.chatbot.llm_provider.get_name())
        else:
            self.status_bar.set_model_info("GPT-2")
        
        # Update MongoDB connection status
        mongodb_connected = hasattr(self.chatbot, 'collection') and self.chatbot.collection is not None
        self.status_bar.set_connection_status(mongodb_connected)

    def add_to_chat_log(self, sender: str, message: str) -> None:
        """Add a message to the chat log.

        Args:
            sender: The sender of the message.
            message: The message content.
        """
        is_user = sender.lower() == "you"
        self.chat_log.add_message(sender, message, is_user)

    def handle_symptoms(self) -> None:
        """Handle the symptoms button click."""
        self.chatbot.chat_mode = "symptoms"
        self.add_to_chat_log("Medical Chatbot", "Please describe your symptoms in detail. I'll ask some follow-up questions to better understand your condition.")

    def handle_emergency(self) -> None:
        """Handle the emergency button click."""
        self.chatbot.chat_mode = "emergency"
        self.add_to_chat_log("Medical Chatbot", "If you are experiencing a medical emergency, please call emergency services (911 in the US) immediately. I can help you find nearby medical facilities.")
        self.chatbot.handle_emergency()

    def handle_general_questions(self) -> None:
        """Handle the general questions button click."""
        self.chatbot.chat_mode = "general"
        self.add_to_chat_log("Medical Chatbot", "You can ask me any questions about medical conditions, treatments, or general health information. How can I help you today?")

    def send_message(self) -> None:
        """Handle sending a message."""
        user_text = self.user_input.text().strip()
        if not user_text:
            return

        self.add_to_chat_log("You", user_text)
        self.user_input.clear()

        if not self.chatbot.chat_mode:
            self.add_to_chat_log("Medical Chatbot", "Please select a conversation mode first using one of the buttons above.")
            return

        # Get response from chatbot
        try:
            if self.chatbot.chat_mode == "symptoms":
                response = self.chatbot.handle_medical_conversation(user_text)
            elif self.chatbot.chat_mode == "emergency":
                response = self.chatbot.generate_response(user_text)
            else:  # general questions
                response = self.chatbot.generate_response(user_text)
                
            self.add_to_chat_log("Medical Chatbot", response)
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            self.add_to_chat_log("Medical Chatbot", "I'm sorry, I encountered an error processing your request. Please try again.")

    def export_conversation(self):
        """Export the current conversation to a file."""
        if not self.chat_log.message_widgets:
            QMessageBox.information(self, "No Conversation", "There is no conversation to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Conversation", "", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(f"Medical Chatbot Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    for widget in self.chat_log.message_widgets:
                        f.write(f"{widget.sender}: {widget.message}\n\n")
                QMessageBox.information(self, "Export Successful", f"Conversation exported to {file_path}")
            except Exception as e:
                logger.error(f"Error exporting conversation: {e}")
                QMessageBox.critical(self, "Export Failed", f"Failed to export conversation: {str(e)}")

    def clear_conversation(self):
        """Clear the current conversation."""
        if not self.chat_log.message_widgets:
            return
            
        reply = QMessageBox.question(
            self, "Clear Conversation", 
            "Are you sure you want to clear the current conversation?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove all message widgets
            for widget in self.chat_log.message_widgets:
                self.chat_log.message_layout.removeWidget(widget)
                widget.deleteLater()
            
            # Clear list
            self.chat_log.message_widgets.clear()
            
            # Reset context
            self.chatbot.context.symptoms.clear()
            self.chatbot.context.responses.clear()
            self.chatbot.context.question_index.clear()
            self.chatbot.context.conversation_history.clear()
            
            # Add welcome message
            self.add_to_chat_log("Medical Chatbot", "Welcome to the Medical Chatbot. How can I help you today?")

    def show_about_dialog(self):
        """Show the about dialog."""
        QMessageBox.about(
            self, "About Medical Chatbot",
            "Medical Chatbot\n\n"
            "A chatbot that provides medical symptom analysis and health guidance.\n\n"
            "Features:\n"
            "- Symptom analysis\n"
            "- Emergency assistance\n"
            "- General medical questions\n\n"
            f"Version: 1.0.0\n"
            f"Using: {self.status_bar.model_info.text()}"
        ) 