"""
=============================================================================
RANDOM FOREST ENSEMBLE TRAINING — Based on Real Multi-Domain NLP Features
=============================================================================
Trains the Random Forest model on the 90 multi-domain corpus samples using
genuine [Perplexity, Burstiness, TTR] statistical features from NLPEngine.
=============================================================================
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from nlp_engine import NLPEngine
from data_corpus import CORPUS

def train_random_forest():
    print(f"Extracting NLP features for {len(CORPUS)} multi-domain samples...")
    nlp = NLPEngine()
    
    X = []
    y = []
    for item in CORPUS:
        features = nlp.extract_features(item["text"])
        X.append(features)
        y.append(item["label"])
        
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training Random Forest on {len(X_train)} samples...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=3, random_state=42)
    clf.fit(X_train, y_train)
    
    val_preds = clf.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"Validation Holdout Accuracy: {acc*100:.1f}%\n")
    print(classification_report(y_val, val_preds, target_names=["HUMAN", "AI", "PARAPHRASED"]))
    
    print("Saving calibrated model to rf_model.pkl...")
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(clf, f)
        
    print("Saved rf_model.pkl successfully.")

if __name__ == "__main__":
    train_random_forest()
