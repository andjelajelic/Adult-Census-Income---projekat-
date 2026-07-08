import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_and_analyze(processed_dir="dataset/processed/", models_dir="models/", results_dir="results/"):
    figures_dir = os.path.join(results_dir, "figures")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Učitavanje test podataka i modela
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).values.ravel()
    
    with open(os.path.join(models_dir, "best_rf_model.pkl"), "rb") as f:
        model = pickle.load(f)
        
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Računanje i čuvanje metrika u TXT fajl [cite: 27]
    metrics_text = (
        f"=== EVALUACIJA MODELA ===\n"
        f"Matrica konfuzije:\n{confusion_matrix(y_test, y_pred)}\n\n"
        f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}\n"
        f"Precision: {precision_score(y_test, y_pred):.4f}\n"
        f"Recall:    {recall_score(y_test, y_pred):.4f}\n"
        f"F1-score:  {f1_score(y_test, y_pred):.4f}\n"
        f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}\n"
    )
    
    with open(os.path.join(metrics_dir, "report.txt"), "w") as f:
        f.write(metrics_text)
    print("[EVALUATE] Metrike su sačuvane u 'results/metrics/report.txt'.")
    
    # Grafički prikaz i čuvanje Feature Importance-a [cite: 28]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances[indices], y=X_test.columns[indices], palette="mako")
    plt.title("Značajnost atributa (Feature Importance)")
    plt.xlabel("Relativna značajnost")
    plt.ylabel("Atributi")
    plt.tight_layout()
    
    # Spasavanje grafika u figures/ 
    plt.savefig(os.path.join(figures_dir, "feature_importance.png"))
    plt.close()
    print("[EVALUATE] Grafici su sačuvani u 'results/figures/'.")

if __name__ == "__main__":
    evaluate_and_analyze()