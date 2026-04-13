import os
import argparse
from dotenv import load_dotenv
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor

def main():
    # 1. Load configuration from .env
    load_dotenv(dotenv_path="config/.env")
    
    # 2. Setup Command Line Arguments
    parser = argparse.ArgumentParser(description="Sherlock Holmes Next-Word Predictor")
    parser.add_argument("--step", choices=["dataprep", "model", "inference", "all"], required=True)
    args = parser.parse_args()

    # Instantiate our tools
    normalizer = Normalizer()

    # --- MILESTONE 1: DATA PREPARATION ---
    if args.step == "dataprep" or args.step == "all":
        print("\n--- Starting Data Preparation ---")
        raw_dir = os.getenv("TRAIN_RAW_DIR")
        output_file = os.getenv("TRAIN_TOKENS")
        
        if not raw_dir or not output_file:
            print("Error: Check your .env for TRAIN_RAW_DIR or TRAIN_TOKENS")
            return

        raw_text = normalizer.load(raw_dir)
        clean_text = normalizer.strip_gutenberg(raw_text)
        sentences = normalizer.sentence_tokenize(clean_text)
        normalizer.save(sentences, output_file)
        print(f"Success! Tokens saved to {output_file}")

    # --- MILESTONE 2: MODEL TRAINING ---
    if args.step == "model" or args.step == "all":
        print("\n--- Starting Model Training ---")
        token_file = os.getenv("TRAIN_TOKENS")
        model_path = os.getenv("MODEL_SAVE_PATH")
        n_value = int(os.getenv("NGRAM_N", 3))

        if not os.path.exists(token_file):
            print(f"Error: Run --step dataprep first.")
            return

        with open(token_file, 'r', encoding='utf-8') as f:
            all_tokens = f.read().split()

        ngram_model = NGramModel(n=n_value)
        ngram_model.train(all_tokens)
        ngram_model.save(model_path)
        print(f"Success! Model saved to {model_path}")

    # --- MILESTONE 3: INFERENCE (PREDICTION) ---
    if args.step == "inference":
        print("\n--- Running Inference Mode ---")
        model_path = os.getenv("MODEL_SAVE_PATH")
        
        if not os.path.exists(model_path):
            print("Error: Model file not found. Run --step model first.")
            return

        predictor = Predictor(model_path)
        
        user_input = input("Enter a starting phrase: ")
        
        # FIXED: Removed 'max_words' to match the new Predictor.generate()
        prediction = predictor.generate(user_input)
        
        print("\n" + "="*40)
        print(f"Input Phrase:  '{user_input}'")
        print(f"Next Word:      {prediction}")
        print("="*40)

if __name__ == "__main__":
    main()