import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. Configuration and Utility Functions
# ==========================================
TRAIN_FILE = 'train_data.json'
TEST_FILE = 'test_data.json'
OUTPUT_FILE = 'baseline_mse.txt'  # New: Result output file path

# Define the five dimensions we will predict
ASPECTS = ['review/appearance', 'review/aroma', 'review/palate', 'review/taste', 'review/overall']

def parse_rating(rating_str):
    """Convert rating strings (e.g., ‘3/5’) to floating-point numbers between 0.0 and 1.0."""
    if not isinstance(rating_str, str):
        return 0.0
    try:
        if '/' in rating_str:
            numerator, denominator = rating_str.split('/')
            return float(numerator) / float(denominator)
        else:
            return float(rating_str)
    except (ValueError, ZeroDivisionError):
        return 0.0

def load_dataset(file_path):
    """Read JSON files and extract text and tags"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    corpus = []
    y_data = {aspect: [] for aspect in ASPECTS}
    
    for entry in data:
        text = entry.get('review/text', '')
        corpus.append(text)
        for aspect in ASPECTS:
            raw_rating = entry.get(aspect, '0/1')
            normalized_score = parse_rating(raw_rating)
            y_data[aspect].append(normalized_score)
            
    return corpus, y_data

# ==========================================
# 2. Main Program
# ==========================================
if __name__ == "__main__":
    print("Loading data...")
    train_corpus, y_train_dict = load_dataset(TRAIN_FILE)
    test_corpus, y_test_dict = load_dataset(TEST_FILE)

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train_corpus)
    X_test = vectorizer.transform(test_corpus)

    print("\nBegin Training and Evaluation (Ridge Regression)...")
    
    # Open the file for writing
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        # Write Table Header
        header = f"{'Aspect':<20} | {'MSE':<20}"
        print(header)
        print("-" * 45)
        f_out.write(header + "\n")
        f_out.write("-" * 45 + "\n")

        for aspect in ASPECTS:
            y_train = y_train_dict[aspect]
            y_test = y_test_dict[aspect]
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            
            # Formatted Output String
            result_str = f"{aspect:<20} | {mse:.5f}"
            
            # Print to console and write to file simultaneously
            print(result_str)
            f_out.write(result_str + "\n")
            
    print(f"\nThe evaluation results have been saved to: {OUTPUT_FILE}")