import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

def generate_synthetic_data(num_samples=2000):
    """
    Generates synthetic data for Perplexity, Burstiness, and TTR.
    Classes: 0 (Human), 1 (AI), 2 (Mixed)
    """
    X = []
    y = []
    
    for _ in range(num_samples):
        label = np.random.choice([0, 1, 2])
        if label == 0: # Human
            ppl = np.random.uniform(50, 600)
            burst = np.random.uniform(20, 100)
            ttr = np.random.uniform(0.6, 1.0)
        elif label == 1: # AI
            ppl = np.random.uniform(5, 45)
            burst = np.random.uniform(0, 15)
            ttr = np.random.uniform(0.3, 0.6)
        else: # Mixed
            ppl = np.random.uniform(30, 80)
            burst = np.random.uniform(10, 30)
            ttr = np.random.uniform(0.5, 0.75)
            
        X.append([ppl, burst, ttr])
        y.append(label)
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Generating synthetic data for Random Forest...")
    X, y = generate_synthetic_data()
    
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X, y)
    
    print("Saving model to rf_model.pkl...")
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(clf, f)
        
    print("Done! Accuracy on synthetic data:", clf.score(X, y))
