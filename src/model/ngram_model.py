import pickle
import os
from collections import defaultdict, Counter

class NGramModel:
    """
    Module Responsibility:
    Calculates probabilities of word sequences. 
    Includes Backoff logic to handle short or unknown prefixes.
    """
    def __init__(self, n=3):
        self.n = n
        # Stores: { (prefix_tuple): {next_word: count} }
        self.model = defaultdict(Counter)

    def train(self, tokens):
        """Trains the model by counting occurrences of n-grams."""
        for i in range(len(tokens) - self.n + 1):
            ngram = tokens[i : i + self.n]
            prefix = tuple(ngram[:-1])
            target = ngram[-1]
            self.model[prefix][target] += 1
        print(f"Training complete. Unique prefixes: {len(self.model)}")

    def predict_next_word(self, prefix_tokens):
        """
        Smarter Prediction with Backoff Logic:
        1. Try the full N-1 prefix (Trigram).
        2. If no match, try the last 1 word (Bigram).
        3. If still no match, return None.
        """
        if not prefix_tokens:
            return None

        # --- Level 1: Trigram Match (Full Prefix) ---
        prefix = tuple(prefix_tokens[-(self.n-1):])
        if prefix in self.model:
            return self.model[prefix].most_common(1)[0][0]

        # --- Level 2: Bigram Backoff (Last 1 Word) ---
        # If we have at least one word, try to find ANY prefix ending in that word
        last_word = prefix_tokens[-1]
        
        # We look for a simpler match in our existing counts
        possible_matches = []
        for p, choices in self.model.items():
            if p[-1] == last_word:
                # Get the most common word for this simplified prefix
                best_word, count = choices.most_common(1)[0]
                possible_matches.append((best_word, count))
        
        if possible_matches:
            # Sort by the highest count and return the best one
            possible_matches.sort(key=lambda x: x[1], reverse=True)
            return possible_matches[0][0]

        return None

    def save(self, filepath):
        """Saves the trained model dictionary to a .pkl file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.model), f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads a model from a .pkl file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls()
        instance.model = data
        return instance