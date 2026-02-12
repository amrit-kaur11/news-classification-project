from src.data_preprocessing import load_and_preprocess
from src.feature_engineering import vectorize
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    
    print("Performing feature engineering...")
    X_train_vec, X_test_vec, _ = vectorize(X_train, X_test)
    
    print("Training model...")
    train_model(X_train_vec, y_train)
    
    print("Evaluating model...")
    accuracy = evaluate_model(X_test_vec, y_test)
    
    print(f"Final Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
