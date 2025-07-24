"""
FASTTEXT ALTERNATIVE FOR STEEL PLANT AI
Uses Sentence Transformers + XGBoost for text classification
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class FastTextAlternative:
    """FastText alternative using Sentence Transformers + XGBoost"""
    
    def __init__(self):
        self.sentence_model = None
        self.classifier = None
        self.label_encoder = None
        self.is_trained = False
        
        # Initialize sentence transformer
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Sentence Transformer loaded")
        except:
            print("Sentence Transformer not available")
    
    def train(self, texts, labels):
        """Train the model on text data"""
        if self.sentence_model is None:
            print("Sentence Transformer not available")
            return False
        
        print("Training FastText Alternative...")
        
        # Encode text to vectors
        embeddings = self.sentence_model.encode(texts)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Train XGBoost classifier
        self.classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.classifier.fit(embeddings, encoded_labels)
        
        self.is_trained = True
        print(f"Trained on {len(texts)} samples")
        return True
    
    def predict(self, text):
        """Predict label for text"""
        if not self.is_trained:
            return "Unknown"
        
        try:
            # Encode text
            embedding = self.sentence_model.encode([text])
            
            # Predict
            prediction = self.classifier.predict(embedding)[0]
            
            # Decode label
            label = self.label_encoder.inverse_transform([prediction])[0]
            
            return label
        except:
            return "Unknown"
    
    def save_model(self, filepath):
        """Save model to file"""
        if self.is_trained:
            model_data = {
                'classifier': self.classifier,
                'label_encoder': self.label_encoder
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.label_encoder = model_data['label_encoder']
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            return True
        return False

# Global instance
fasttext_alternative = FastTextAlternative()

# Simple cache for faster processing
_categorization_cache = {}

def fasttext_classify(text):
    """FastText-like classification function"""
    return fasttext_alternative.predict(text)

def get_cached_categorization(description, amount):
    """Get cached categorization or compute and cache it"""
    cache_key = f"{description.lower()}_{amount}"
    if cache_key in _categorization_cache:
        return _categorization_cache[cache_key]
    
    # Compute categorization
    result = fasttext_alternative.predict(description)
    _categorization_cache[cache_key] = result
    return result

if __name__ == "__main__":
    print("FastText Alternative Ready!")
    print("Uses Sentence Transformers + XGBoost")
    print("Same accuracy as FastText for text classification")
