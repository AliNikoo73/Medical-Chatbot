# 🩺 **Medical Chatbot for Symptom Analysis and Disease Guidance**

---

## 📜 **Summary**

This project develops a local, AI-driven **medical chatbot** designed to assist users with **symptom analysis** and provide preliminary **health guidance**. The chatbot employs a **pre-trained GPT-2 model** for natural language understanding and integrates with **MongoDB** to store user interactions.

Key features include:
- **Symptom-based analysis**: Extract relevant medical conditions and offer follow-up questions.
- **Medical recommendations**: Provide tailored guidance based on a pre-programmed symptom questionnaire.
- **Emergency response**: Direct users to nearby medical facilities in urgent cases.
  
The chatbot includes a **Graphical User Interface (GUI)** built with **PyQt5**, enabling easy interaction with users. The project demonstrates the integration of **machine learning** for conversational AI, **symptom classification**, and basic medical guidance, while focusing on a **user-friendly interface**.

---

## 🎯 **Objective**
> To develop a medical chatbot that helps users by analyzing symptoms and providing tailored health-related recommendations and emergency guidance.

---

## 🛠 **Skills Required**

### **Technical Skills**

![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/-Natural%20Language%20Processing-007396?style=for-the-badge&logo=nlp&logoColor=white)
![GPT-2](https://img.shields.io/badge/-GPT--2-4A90E2?style=for-the-badge&logo=OpenAI&logoColor=white)
![PyQt5](https://img.shields.io/badge/-PyQt5-41CD52?style=for-the-badge&logo=qt&logoColor=white)
![MongoDB](https://img.shields.io/badge/-MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Torch](https://img.shields.io/badge/-Torch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

- **Python programming**
- **Natural Language Processing (NLP)** using GPT-2
- **PyQt5** for GUI development
- **MongoDB** for data storage and retrieval
- **Torch** for machine learning model execution
- **Basic understanding of medical terminologies**

### **Soft Skills**

- 🧠 **Problem-solving**
- 🎯 **Attention to Detail**
- 🎨 **User Interface Design**
- 🗣️ **Communication** for handling user input effectively

---

## 📊 **Deliverables**

### **Key Outputs**

- 🤖 **A functional chatbot application** with:
  - Symptom-based conversation flow
  - Dynamic question generation based on user input
  - Real-time medical recommendations
  - A fully operational **GUI** for user interaction
- 🧠 **Pre-trained NLP model integration** for text generation and intent recognition
- 🗂 **Stored conversation logs** in MongoDB for future analysis or improvements
- 🚨 **Emergency handling system** with integrated browser redirection for urgent care

---

## 🔍 **Additional Information**

- **Model Used**: **GPT-2** for generating dynamic responses.
- **Conversation Flow**: Customized using a **keyword-based intent extraction system**.
- **Data Storage**: MongoDB is used to maintain user-specific context and manage multiple symptoms.
- **GUI Development**: **PyQt5** ensures ease of use and a professional interface.
- **Environment**: Built locally to provide quick access and ensure user **privacy** during conversations.

## Project Structure

```
medical-chatbot/
├── src/
│   ├── main.py              # Application entry point
│   ├── chatbot/
│   │   └── core.py         # Core chatbot functionality
│   ├── gui/
│   │   └── main_window.py  # GUI implementation
│   └── database/
│       └── mongodb.py      # Database operations
├── config/
│   └── settings.py         # Configuration settings
└── requirements/
    ├── base.txt           # Core dependencies
    └── dev.txt            # Development dependencies
```

## Prerequisites

- Python 3.8 or higher
- MongoDB
- Virtual environment (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AliNikoo73/Medical-Chatbot.git
   cd Medical-Chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements/base.txt
   ```

4. Start MongoDB:
   ```bash
   # On macOS with Homebrew:
   brew services start mongodb-community
   ```

## Running the Application

1. Ensure MongoDB is running
2. Run the application:
   ```bash
   PYTHONPATH=$PYTHONPATH:. python src/main.py
   ```

## Usage

1. Launch the application
2. Choose a conversation mode:
   - "I am not feeling good!!" for symptom analysis
   - "I need a doctor right away" for emergency help
   - "I want to ask questions about my disease" for general questions
3. Type your message in the input field
4. Press Enter or click Send to interact with the chatbot

## Features in Detail

### Symptom Analysis
- Interactive questioning about symptoms
- Follow-up questions based on responses
- Medical recommendations based on symptoms

### Emergency Assistance
- Quick access to emergency resources
- Maps integration for nearby clinics
- Emergency guidance

### General Medical Questions
- AI-powered responses using GPT-2
- Context-aware conversations
- Medical information retrieval

### Data Storage
- MongoDB integration for conversation history
- Secure storage of medical interactions
- Easy retrieval of past conversations

## Development

For development setup, install additional dependencies:
```bash
pip install -r requirements/dev.txt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

