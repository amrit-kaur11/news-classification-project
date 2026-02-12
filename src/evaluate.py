from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
from src.config import MODEL_PATH, METRICS_PATH

def evaluate_model(X_test, y_test):
    os.makedirs("results", exist_ok = True)
    model = joblib.load(MODEL_PATH)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    with open(METRICS_PATH, "w") as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(str(cm))
    
    return acc
