from sklearn.linear_model import LogisticRegression
import joblib
import os
from src.config import MODEL_PATH

def train_model(X_train, y_train):
    os.makedirs("models", exist_ok = True)
    model = LogisticRegression(max_iter=2000,C=1.5,solver='lbfgs')
    model.fit(X_train, y_train)
    
    joblib.dump(model, MODEL_PATH)
    return model
