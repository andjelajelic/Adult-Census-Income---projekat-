import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(model, X_test, y_test, model_name):
    """Računa standardne klasifikacione metrike za dati model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    return metrics


def format_metrics_block(metrics):
    return (
        f"--- {metrics['model']} ---\n"
        f"Matrica konfuzije:\n{metrics['confusion_matrix']}\n\n"
        f"Accuracy:  {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall:    {metrics['recall']:.4f}\n"
        f"F1-score:  {metrics['f1']:.4f}\n"
        f"ROC-AUC:   {metrics['roc_auc']:.4f}\n"
    )


def plot_model_comparison(all_metrics, figures_dir):
    """Bar-chart poređenje modela po svim metrikama."""
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comparison_df = pd.DataFrame(
        [[m[name] for name in metric_names] for m in all_metrics],
        index=[m["model"] for m in all_metrics],
        columns=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    )

    ax = comparison_df.T.plot(kind="bar", figsize=(10, 5), colormap="mako_r")
    ax.set_title("Poređenje performansi modela")
    ax.set_ylabel("Vrednost metrike")
    ax.set_ylim(0, 1)
    ax.legend(title="Model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_comparison.png"))
    plt.close()

    return comparison_df


def evaluate_and_analyze(processed_dir="dataset/processed/", models_dir="models/", results_dir="results/"):
    figures_dir = os.path.join(results_dir, "figures")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Učitavanje test podataka
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).values.ravel()

    # Učitavanje oba modela - baznog (Logistic Regression) i optimizovanog (Random Forest)
    with open(os.path.join(models_dir, "logistic_model.pkl"), "rb") as f:
        log_model = pickle.load(f)
    with open(os.path.join(models_dir, "best_rf_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)

    log_metrics = compute_metrics(log_model, X_test, y_test, "Logistic Regression")
    rf_metrics = compute_metrics(rf_model, X_test, y_test, "Random Forest")
    all_metrics = [log_metrics, rf_metrics]

    # Računanje i čuvanje metrika oba modela u TXT fajl [cite: 27]
    report_text = "=== EVALUACIJA I POREĐENJE MODELA ===\n\n"
    report_text += format_metrics_block(log_metrics) + "\n"
    report_text += format_metrics_block(rf_metrics) + "\n"

    best_model_name = max(all_metrics, key=lambda m: m["f1"])["model"]
    report_text += (
        f"=== ZAKLJUČAK ===\n"
        f"Model sa boljim F1-skorom (balans preciznosti i odziva): {best_model_name}\n"
    )

    with open(os.path.join(metrics_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    print("[EVALUATE] Metrike oba modela su sačuvane u 'results/metrics/report.txt'.")

    # Grafičko poređenje modela
    comparison_df = plot_model_comparison(all_metrics, figures_dir)
    comparison_df.to_csv(os.path.join(metrics_dir, "model_comparison.csv"))
    print("[EVALUATE] Poređenje modela sačuvano u 'results/figures/model_comparison.png' i 'results/metrics/model_comparison.csv'.")

    # Grafički prikaz i čuvanje Feature Importance-a za RF (jedini od dva modela koji ovo direktno podržava) [cite: 28]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importances[indices], y=X_test.columns[indices], hue=X_test.columns[indices], palette="mako", legend=False)
    plt.title("Značajnost atributa (Feature Importance) - Random Forest")
    plt.xlabel("Relativna značajnost")
    plt.ylabel("Atributi")
    plt.tight_layout()
    
    # Spasavanje grafika u figures/ 
    plt.savefig(os.path.join(figures_dir, "feature_importance.png"))
    plt.close()
    print("[EVALUATE] Grafici su sačuvani u 'results/figures/'.")

if __name__ == "__main__":
    evaluate_and_analyze()