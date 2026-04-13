import os
import re
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK data (required for tokenization)
nltk.download('punkt', quiet=True)

class Normalizer:
    """
    Module Responsibility:
    Handles loading, cleaning, stripping, and tokenizing the Sherlock Holmes corpus.
    Ensures consistent text processing for both training and inference.
    """

    def load(self, folder_path):
        """Loads all .txt files from a folder and returns them as a single string."""
        combined_text = ""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
            
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    combined_text += f.read() + " "
        return combined_text

    def strip_gutenberg(self, text):
        """Removes Project Gutenberg headers and footers using standard markers."""
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        
        # Find the end of the start marker line
        start_idx = text.find(start_marker)
        if start_idx != -1:
            # Move index to the end of that specific marker line
            line_end = text.find("***", start_idx + len(start_marker))
            text = text[line_end + 3:]
            
        # Find the beginning of the end marker
        end_idx = text.find(end_marker)
        if end_idx != -1:
            text = text[:end_idx]
            
        return text.strip()

    def normalize(self, text):
        """
        Applies all normalization steps in the strict required order:
        lowercase -> remove punctuation -> remove numbers -> remove whitespace.
        """
        # 1. Lowercase
        text = text.lower()
        # 2. Remove Punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # 3. Remove Numbers
        text = re.sub(r'\d+', '', text)
        # 4. Remove Whitespace (extra spaces and newlines)
        text = " ".join(text.split())
        
        return text

    def sentence_tokenize(self, text):
        """Splits raw text into a list of sentences."""
        return sent_tokenize(text)

    def word_tokenize(self, sentence):
        """Splits a single sentence into a list of space-separated tokens."""
        return word_tokenize(sentence)

    def save(self, sentences, filepath):
        """Writes tokenized sentences to the output file (one sentence per line)."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                # Clean each sentence and join tokens with spaces
                clean_sentence = self.normalize(sentence)
                if clean_sentence:  # Don't save empty lines
                    f.write(clean_sentence + "\n")