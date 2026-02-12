# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 22:26:09 2026

@author: wilso
"""
# Importing relevant libraries
import json
import random
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class ChatBot:
    
    # Exit commands
    exit_words = ["exit", "quit", "stop", "bye", "goodbye"]
    
    def __init__(self):
        # Loading intents and responses from the json file
        with open("intents.json") as file:
            data = json.load(file)

        self.intents = data["intents"]
        self.responses = data["responses"]
    
        # Train word2vec
        self.train_word2vec()
        
        # Build TF-IDF + Logistic Regression model
        self.vectorizer, self.model = self.vec_model(self.intents)
       
    def suggest_related_terms(self, word):
        if word in self.w2v_model.wv:
            similar = self.w2v_model.wv.most_similar(word, topn=3)
            return [w for w, _ in similar]
        return []
    
    # word2vec model
    def train_word2vec(self):
        sentences = []
    
        for examples in self.intents.values():
            for sentence in examples:
                tokens = sentence.lower().split()
                sentences.append(tokens)
    
        self.w2v_model = Word2Vec(
            sentences,
            vector_size=100,
            window=3,
            min_count=1,
            workers=4
        )
    
    # Apply word2vec
    def vec_model(self, intents):
        sentences = []
        labels = []
    
        for intent, examples in intents.items():
            for sentence in examples:
                sentences.append(sentence.lower())
                labels.append(intent)
    
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1,2)
        )
    
        X = vectorizer.fit_transform(sentences)
    
        model = LogisticRegression(max_iter=1000)
        model.fit(X, labels)
    
        return vectorizer, model
    
    def end_chat(self, reply):
        # Checking if the intent is to exit the conversation
        for command in self.exit_words:
            if command in reply:
                return True
        return False

    def extract_entities(self, reply):
        """
        Extract entities from user input based on a keyword-to-entity mapping.
        Returns a list of tuples: (entity_name, "KEYWORD")
        """
        reply = reply.lower()  # normalize input
        entity_map = {
            "int": "int",
            "integer": "int",
            "float": "float",
            "str": "str",
            "string": "str",
            "list": "list",
            "tuple": "tuple",
            "dictionary": "dict",
            "dict": "dict",
            "bool": "bool",
            "for loop": "for",
            "while loop": "while"
        }
    
        found = []
        for word, entity in entity_map.items():
            if word in reply:
                found.append((entity, "KEYWORD"))
    
        return found

    
    def get_response(self, intent, entities):
        resp = self.responses[intent]
    
        # Simple list of responses
        if isinstance(resp, list):
            return random.choice(resp)
    
        # Nested dict for entity-aware responses
        elif isinstance(resp, dict):
            for ent in entities:
                key = ent[0].lower()
                if key in resp:
                    if isinstance(resp[key], list):
                        return random.choice(resp[key])
                    return resp[key]
            # fallback to default
            if "default" in resp:
                return random.choice(resp["default"])
        
        return "Sorry, I don't understand."

    def chat(self):
        # Starting the conversation
        reply = input("Welcome, I'm Pyla, Python Learning Assistant, how can I be of use?\n").lower()
        
        # Keep conversation going
        while not self.end_chat(reply):
            reply_vec = self.vectorizer.transform([reply])
            intent = self.model.predict(reply_vec)[0]
                
            # Confidence check
            probs = self.model.predict_proba(reply_vec)
            confidence = max(probs[0])
            
            #print("Confidence:", confidence)

            if confidence < 0.2:
                reply = input("Pyla: I'm not sure I understood that. Could you rephrase?\n").lower()
                continue

            # Extract entities
            entities = self.extract_entities(reply)
            
            # Suggest related terms
            for entity_word, _ in entities:
                related = self.suggest_related_terms(entity_word)
                if related:
                    print(f"Pyla: Related concepts: {', '.join(related)}")

            
            # Get safe response
            bot_reply = self.get_response(intent, entities)
            reply = input(f"Pyla: {bot_reply}").lower()
        
        print("Pyla: Goodbye! Happy coding!")

# Instanciating the ChatBot
Pyla = ChatBot()
Pyla.chat()
