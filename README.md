# Pyla: Python Learning Assistant
**Author:** Wilson Inacio
**Codedcademy Username:** @Bezeus

## Project Overview
Pyla is a retrieval-based chatbot designed to assist users in learning Python programming. Users can ask questions about Python basics, variables, data types, loops, functions, and common errors, and Pyla will provide contextually relevant answers.

The purpose of this project is to showcase the application of **natural language processing (NLP)** and **machine learning (ML)** techniques in building a closed-domain educational chatbot.

## Chatbot
**Pyla** - Retrieval-based chatbot using a closed-domain architecture:
- **Intent classification:** TF-IDF vectorization of user input combined with a Logistic Regression classifier to predict the user’s intent.
- **Entity recognition:** A keyword-to-entity mapping to identify specific programming concepts mentioned in user input.
- **Response selection:** Predefined responses stored in intents.json, returned based on predicted intent and recognized entities.

## Use Cases
Pyla can help begginers learning Python to quickly get explanations of key conceptes, and clarify differences between similar concepts (e.g., `list` vs `tuple`).

## Techniques Used
- **TF-IDF vectorization** (`sklearn.featuer_extraction.text.TfidfVectorizer`) to transform user input into numerical features.
- **Logistic Regression** (`sklearn.linear_model.LogisticRegression`) for intent classification.
- **Keyword based entity recognition** for context-aware responses.
- **Randomized response selection** for varied and natural conversation.

## Installation & Dependencies
**Required libraries**
- `json`
- `random`
- `sklearn` (TF-IDF, Logistic Regression)

**Files Included**
- `chatbot.py` - main chatbot implementation
- `intents.json` - dataset of intents, example phrases and responses

## How to run
```python
python chatbot.py
```
Sample interaction:
```python
Welcome, I'm Pyla, Python Learning Assistant, how can I be of use?
You: what is python?
Pyla: Python is a versatile programming language known for its readability.

You: list vs tuple
Pyla: Lists are ordered, mutable collections of items. Tuples are ordered, immutable collections.

You: exit # exit, quit or bye to end chat
```

## Reflection
**Challenges faced:**
- Balancing intent coverage while avoiding overfitting to exact phrases.
- Implementing entity-specific responses to handle Python concepts dynamically.
**Successes:**
- Pyla correctly identifies intents and provides meaningful answers.
- Closed-domain design ensures reliable, accurate responses.
**Learnings:**
- TD-IDF combined with Logistic Regression is a simple yet effective approach for intent classification.
- Keyword-based entity recognition can be enough for educational chatbots when data is limited.
**Ethical Concerns:**
- The chatbot only provides predefined responses and cannot verify user code correctness
- Users should not rely on Pyla for critical programming advice.

**Visual Demo**
![Screenshot](/Screenshots/screenshot.png)
