import os
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_models(processed_dir="dataset/processed/", models_dir="models/"):
    os.makedirs(models_dir, exist_ok=True)
    
    # Učitavanje procesiranih podataka
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()
    
    print("[TRAIN] Treniranje baznog modela (Logistic Regression)...")
    log_model = LogisticRegression(max_iter=10000, solver="liblinear")
    log_model.fit(X_train, y_train)
    with open(os.path.join(models_dir, "logistic_model.pkl"), "wb") as f:
        pickle.dump(log_model, f)
        
    print("[TRAIN] Optimizacija hiperparametara za Random Forest...")
    param_grid = {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_estimators=50), 
        param_grid=param_grid, 
        cv=3, 
        scoring='f1'
    )
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"[TRAIN] Najbolji parametri za RF: {grid_search.best_params_}")
    
    # Čuvanje najboljeg modela za deployment
    with open(os.path.join(models_dir, "best_rf_model.pkl"), "wb") as f:
        pickle.dump(best_rf, f)
        
    print("[TRAIN] Modeli su uspešno sačuvani u 'models/'.")

if __name__ == "__main__":
    train_models()