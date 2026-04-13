import os
from src.model.ngram_model import NGramModel

class Predictor:
    """
    Module Responsibility:
    Loads a trained N-Gram model and provides a simple interface 
    to predict the single next word in a sequence.
    """
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")
        
        # Load the 'brain' we trained in Milestone 2
        self.model = NGramModel.load(model_path)

    def generate(self, seed_text):
        """
        Takes a string of text and returns ONLY the 
        one most probable next word.
        """
        # 1. Clean the input to match our training data (lowercase)
        tokens = seed_text.lower().split()
        
        if not tokens:
            return "(Please enter a phrase)"

        # 2. Ask the model for the next word using its Backoff logic
        next_word = self.model.predict_next_word(tokens)
        
        # 3. Return the single word result
        if next_word:
            return next_word
        else:
            return "(No prediction found in Sherlock's vocabulary)"