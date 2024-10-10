import sys
import re
import webbrowser
from pymongo import MongoClient
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt

class LocalChatbot:
    def __init__(self, model_name='gpt2'):
        print(f"Loading {model_name}, this may take a few minutes...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        print("Model loaded successfully!")

        self.client = MongoClient('mongodb://localhost:27017/')
        self.collection = self.client['clinic_chatbot']['conversations']
        self.context = {'symptoms': [], 'responses': {}, 'question_index': {}}
        self.chat_mode = None

    def extract_intents_and_entities(self, text):
        medical_keywords = [
            'headache', 'fever', 'pain', 'cough', 'diabetes', 'asthma', 'flu', 'allergy',
            'fatigue', 'dizziness', 'nausea', 'hypertension', 'cholesterol', 'cancer', 'tumor', 'infection',
            'virus', 'seizure', 'arthritis', 'depression', 'anxiety', 'migraine', 'stroke', 'heart attack'
        ]
        return [word for word in medical_keywords if re.search(rf'\b{word}\b', text, re.IGNORECASE)]

    def handle_medical_conversation(self, user_text):
        if entities := self.extract_intents_and_entities(user_text):
            new_symptom = entities[0]
            if new_symptom not in self.context['symptoms']:
                self.context['symptoms'].append(new_symptom)
                self.context['responses'][new_symptom], self.context['question_index'][new_symptom] = [], 0
                self.save_conversation(new_symptom, "start", "New symptom conversation started.")
            return self.ask_question(new_symptom)

        for symptom in self.context['symptoms']:
            if self.context['question_index'][symptom] >= len(self.get_symptom_questions(symptom)):
                return self.provide_recommendation(symptom)

            self.context['responses'][symptom].append(user_text)
            self.save_conversation(symptom, "user", user_text)
            self.context['question_index'][symptom] += 1
            return self.ask_question(symptom)
        return self.generate_response(user_text)

    def ask_question(self, symptom):
        questions = self.get_symptom_questions(symptom)
        index = self.context['question_index'][symptom]
        question = questions[index] if index < len(questions) else self.provide_recommendation(symptom)
        self.save_conversation(symptom, "bot", question)
        return question

    def get_symptom_questions(self, symptom):
        questions_bank = {
            'headache': ["Can you describe the severity of your headache?", "When did it start?", "Is it continuous or intermittent?"],
            'fever': ["What was your last body temperature?", "When did the fever start?", "Do you have chills or sweats?"],
            'diabetes': ["How often do you monitor blood sugar?", "Do you have excessive thirst or urination?"],
            'asthma': ["How often do you experience shortness of breath?", "Are symptoms triggered by exercise, allergens, etc.?"],
            'flu': ["When did your symptoms start?", "Do you have respiratory symptoms like cough, sore throat?"],
            'allergy': ["What symptoms do you experience?", "Are your symptoms seasonal or year-round?"]
            # Add more symptoms and questions as needed...
        }
        return questions_bank.get(symptom, ["Please provide more details about your symptom."])

    def provide_recommendation(self, symptom):
        recommendations = {
            'headache': "Rest, stay hydrated, and avoid bright lights. If it persists, consult a doctor.",
            'fever': "Monitor your temperature and stay hydrated. If high or persistent, seek medical help."
        }
        recommendation = recommendations.get(symptom, "Consult with a doctor for further assessment.")
        self.save_conversation(symptom, "bot", recommendation)
        return recommendation

    def generate_response(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(
            input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.95,
            temperature=0.7, pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def save_conversation(self, symptom, role, message):
        self.collection.insert_one({'symptom': symptom, 'role': role, 'message': message})

    def handle_emergency(self):
        print("Emergency detected. Please call 911 or seek urgent medical care.")
        webbrowser.open("https://www.google.com/maps/search/clinic+near+me/")

class ChatbotGUI(QMainWindow):
    def __init__(self, chatbot):
        super().__init__()
        self.chatbot = chatbot
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Clinic Chatbot")
        self.setGeometry(100, 100, 600, 500)
        layout = QVBoxLayout()

        self.chat_log = QTextEdit(self)
        self.chat_log.setReadOnly(True)
        layout.addWidget(self.chat_log)

        for label, handler in [
            ("I am not feeling good!!", self.handle_symptoms),
            ("I need a doctor right away", self.handle_emergency),
            ("I want to ask questions about my disease", self.handle_general_questions)
        ]:
            button = QPushButton(label, self)
            button.clicked.connect(handler)
            layout.addWidget(button)

        self.user_input = QLineEdit(self)
        self.user_input.returnPressed.connect(self.send_message)
        layout.addWidget(self.user_input)

        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.send_message)
        layout.addWidget(self.send_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def handle_symptoms(self):
        self.chatbot.chat_mode = "symptoms"
        self.add_to_chat_log("Chatbot", "Please describe your symptoms.")

    def handle_emergency(self):
        self.chatbot.chat_mode = "emergency"
        self.add_to_chat_log("Chatbot", "Please call 911 or seek emergency care.")
        self.chatbot.handle_emergency()

    def handle_general_questions(self):
        self.chatbot.chat_mode = "general"
        self.add_to_chat_log("Chatbot", "You can ask me any questions about your disease.")

    def add_to_chat_log(self, sender, message):
        self.chat_log.append(f"{sender}: {message}")

    def send_message(self):
        user_text = self.user_input.text().strip()
        self.add_to_chat_log("You", user_text)
        self.user_input.clear()
        response = self.chatbot.handle_medical_conversation(user_text) if self.chatbot.chat_mode == "symptoms" else "I'm here to help!"
        self.add_to_chat_log("Chatbot", response)

if __name__ == "__main__":
    chatbot = LocalChatbot()
    app = QApplication(sys.argv)
    gui = ChatbotGUI(chatbot)
    gui.show()
    sys.exit(app.exec_())