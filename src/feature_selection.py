import os
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(model, X_test, y_test, model_name):
    """Računa standardne klasifikacione metrike za dati model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def select_top_features_rfe(X_train, y_train, top_n=6):
    """
    WRAPPER metoda odabira atributa - Recursive Feature Elimination (RFE).

    Za razliku od embedded pristupa (npr. feature_importances_ direktno iz jednog
    istreniranog modela), RFE je wrapper metoda: iterativno trenira model, uklanja
    trenutno najslabiji atribut na osnovu njegovog doprinosa performansama, pa
    trenira ponovo - i tako dok ne ostane tačno top_n atributa. Kriterijum za
    eliminaciju je stvarni doprinos modelu kroz ponovljeno treniranje, a ne samo
    statistika iz jednog prolaza.
    """
    estimator = RandomForestClassifier(random_state=42, n_estimators=50)
    selector = RFE(estimator=estimator, n_features_to_select=top_n, step=1)
    selector = selector.fit(X_train, y_train)

    top_features = list(X_train.columns[selector.support_])
    # Redosled eliminacije (ranking_ == 1 znači da je atribut ostao do kraja)
    ranking = pd.Series(selector.ranking_, index=X_train.columns).sort_values()
    return top_features, ranking


def plot_feature_comparison(all_metrics, figures_dir):
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comparison_df = pd.DataFrame(
        [[m[name] for name in metric_names] for m in all_metrics],
        index=[m["model"] for m in all_metrics],
        columns=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    )

    ax = comparison_df.T.plot(kind="bar", figsize=(10, 5), colormap="mako_r")
    ax.set_title("Random Forest: svi atributi vs. najznačajniji atributi")
    ax.set_ylabel("Vrednost metrike")
    ax.set_ylim(0, 1)
    ax.legend(title="Model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "feature_selection_comparison.png"))
    plt.close()

    return comparison_df


def run_feature_selection(processed_dir="dataset/processed/", models_dir="models/",
                           results_dir="results/", top_n=6):
    figures_dir = os.path.join(results_dir, "figures")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Učitavanje procesiranih podataka
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).values.ravel()

    # Učitavanje već istreniranog RF modela (sa svim atributima) - koristi se za rangiranje značajnosti
    with open(os.path.join(models_dir, "best_rf_model.pkl"), "rb") as f:
        full_model = pickle.load(f)

    # 1. WRAPPER odabir top_n najznačajnijih atributa pomoću RFE
    print(f"[FEATURE SELECTION] Pokrećem RFE (wrapper metoda) - ovo traje duže jer se model trenira iterativno...")
    start_time = time.time()
    top_features, ranking = select_top_features_rfe(X_train, y_train, top_n=top_n)
    rfe_duration = time.time() - start_time
    print(f"[FEATURE SELECTION] RFE završen za {rfe_duration:.1f}s. Top {top_n} atributa: {top_features}")

    # 2. Treniranje novog RF modela SAMO sa top atributima (isti hiperparametri kao pun model,
    #    radi fer poređenja doprinosa selekcije atributa, a ne doprinosa tuninga)
    X_train_top = X_train[top_features]
    X_test_top = X_test[top_features]

    reduced_model = RandomForestClassifier(
        random_state=42,
        n_estimators=full_model.n_estimators,
        max_depth=full_model.max_depth,
        min_samples_split=full_model.min_samples_split,
    )
    reduced_model.fit(X_train_top, y_train)

    # 3. Evaluacija oba modela na test setu
    full_metrics = compute_metrics(full_model, X_test, y_test, f"RF - svi atributi ({X_train.shape[1]})")
    reduced_metrics = compute_metrics(reduced_model, X_test_top, y_test, f"RF - top {top_n} atributa")
    all_metrics = [full_metrics, reduced_metrics]

    # 4. Čuvanje izveštaja
    report_lines = ["=== ODABIR NAJZNAČAJNIJIH ATRIBUTA (WRAPPER METODA - RFE) ===\n"]
    report_lines.append(
        "Metodologija: Recursive Feature Elimination (RFE) je wrapper metoda odabira atributa.\n"
        "Za razliku od filter metoda (koje gledaju samo statističku vezu atributa i cilja,\n"
        "npr. korelaciju) ili embedded metoda (gde se značajnost čita direktno iz jednog već\n"
        "istreniranog modela), RFE iterativno trenira model, uklanja atribut sa najmanjim\n"
        "doprinosom, pa trenira ponovo - sve dok ne ostane željeni broj atributa. Ovo je\n"
        "skuplje računski (model se trenira više puta), ali odabir je direktno vođen\n"
        f"stvarnim uticajem atributa na performanse modela.\n"
        f"Vreme izvršavanja RFE-a: {rfe_duration:.1f}s (estimator: RandomForestClassifier, n_estimators=50)\n"
    )
    report_lines.append(f"Ukupan broj atributa u punom modelu: {X_train.shape[1]}")
    report_lines.append(f"Odabrano top {top_n} atributa (RFE, estimator=RandomForest):")
    for feat in top_features:
        report_lines.append(f"  - {feat}")

    report_lines.append(f"\nKompletno rangiranje svih atributa po RFE eliminaciji (1 = ostao u top {top_n}, veći broj = ranije eliminisan):")
    for feat, rank in ranking.items():
        report_lines.append(f"  {rank}. {feat}")

    report_lines.append("\n=== POREĐENJE PERFORMANSI ===\n")
    for m in all_metrics:
        report_lines.append(
            f"--- {m['model']} ---\n"
            f"Accuracy:  {m['accuracy']:.4f}\n"
            f"Precision: {m['precision']:.4f}\n"
            f"Recall:    {m['recall']:.4f}\n"
            f"F1-score:  {m['f1']:.4f}\n"
            f"ROC-AUC:   {m['roc_auc']:.4f}\n"
        )

    f1_diff = full_metrics["f1"] - reduced_metrics["f1"]
    report_lines.append("=== ZAKLJUČAK ===")
    if abs(f1_diff) < 0.01:
        report_lines.append(
            f"Razlika u F1-skoru je zanemarljiva ({f1_diff:+.4f}) - model sa top {top_n} atributa "
            f"postiže gotovo identične performanse kao model sa svim atributima, uz manju "
            f"kompleksnost i brže treniranje/inferenciju. Preporučuje se model sa redukovanim "
            f"skupom atributa za produkcionu upotrebu."
        )
    elif f1_diff > 0:
        report_lines.append(
            f"Pun model je bolji za {f1_diff:.4f} F1-poena. Preostali atributi i dalje nose "
            f"korisnu prediktivnu informaciju, pa uklanjanje nije opravdano ako je prioritet "
            f"maksimalna tačnost, ali redukovani model ostaje razumna opcija ako je interpretabilnost "
            f"ili brzina važnija od poslednjeg procenta performansi."
        )
    else:
        report_lines.append(
            f"Redukovani model je čak bolji za {-f1_diff:.4f} F1-poena, što ukazuje da su neki od "
            f"izostavljenih atributa uneli šum/redundansu. Preporučuje se model sa top {top_n} atributa."
        )

    with open(os.path.join(metrics_dir, "feature_selection_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("[FEATURE SELECTION] Izveštaj sačuvan u 'results/metrics/feature_selection_report.txt'.")

    # 5. Grafičko poređenje
    comparison_df = plot_feature_comparison(all_metrics, figures_dir)
    comparison_df.to_csv(os.path.join(metrics_dir, "feature_selection_comparison.csv"))
    print("[FEATURE SELECTION] Grafik sačuvan u 'results/figures/feature_selection_comparison.png'.")

    # 6. Čuvanje redukovanog modela kao dodatnog artefakta za deployment (opciono, brži model)
    with open(os.path.join(models_dir, "best_rf_model_top_features.pkl"), "wb") as f:
        pickle.dump(reduced_model, f)
    with open(os.path.join(models_dir, "top_features.pkl"), "wb") as f:
        pickle.dump(top_features, f)
    print(f"[FEATURE SELECTION] Redukovani model sačuvan u '{models_dir}best_rf_model_top_features.pkl'.")


if __name__ == "__main__":
    run_feature_selection()
