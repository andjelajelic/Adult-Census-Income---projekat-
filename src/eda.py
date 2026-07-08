import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

NUMERIC_COLS = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                 'capital-loss', 'hours-per-week']

CATEGORICAL_COLS = ['workclass', 'education', 'marital-status', 'occupation',
                     'relationship', 'race', 'sex', 'native-country']


def load_and_clean(train_path, test_path):
    """
    Učitava sirove podatke i radi MINIMALNO čišćenje (bez enkodiranja) - isto
    kao prvi koraci u data_preparation.py - kako bi EDA bila urađena na
    čitljivim, ali validnim podacima.
    """
    df_train = pd.read_csv(train_path, header=None, names=COLUMN_NAMES, on_bad_lines='skip')
    df_test = pd.read_csv(test_path, header=None, names=COLUMN_NAMES, on_bad_lines='skip')

    # Uklanjanje redova sa tekstualnim šumom (npr. '|1x3 Cross validator')
    df_train['age'] = pd.to_numeric(df_train['age'], errors='coerce')
    df_test['age'] = pd.to_numeric(df_test['age'], errors='coerce')
    df_train = df_train.dropna(subset=['age'])
    df_test = df_test.dropna(subset=['age'])

    df = pd.concat([df_train, df_test], ignore_index=True)

    # Trim razmaka u tekstualnim kolonama
    obj_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # '?' -> NaN
    df = df.replace('?', np.nan)

    # income ima '.' na kraju u test setu (>50K.) - normalizujemo za čitljivost
    df['income'] = df['income'].str.replace('.', '', regex=False)

    return df


def report_missing_and_anomalies(df, report_lines):
    report_lines.append("=== NEDOSTAJUĆE VREDNOSTI ===")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        report_lines.append("Nema nedostajućih vrednosti nakon zamene '?' -> NaN.")
    else:
        for col, cnt in missing.items():
            pct = 100 * cnt / len(df)
            report_lines.append(f"  {col}: {cnt} ({pct:.2f}%)")

    report_lines.append("\n=== OSNOVNE STATISTIKE NUMERIČKIH ATRIBUTA ===")
    report_lines.append(df[NUMERIC_COLS].describe().round(2).to_string())

    report_lines.append("\n=== POTENCIJALNE ANOMALIJE (na osnovu domenskog znanja) ===")
    # capital-gain / capital-loss su skoro uvek 0 (retki eventi)
    for col in ['capital-gain', 'capital-loss']:
        zero_pct = 100 * (df[col] == 0).mean()
        report_lines.append(f"  {col}: {zero_pct:.1f}% redova ima vrednost 0 (izrazito retke nenulte vrednosti - očekivano za ovu vrstu atributa)")

    # hours-per-week ekstremi
    low_hours = (df['hours-per-week'] < 5).sum()
    high_hours = (df['hours-per-week'] > 80).sum()
    report_lines.append(f"  hours-per-week: {low_hours} redova sa < 5h nedeljno, {high_hours} redova sa > 80h nedeljno (proveriti da li su realni podaci ili greška unosa)")

    # age ekstremi
    report_lines.append(f"  age: opseg {df['age'].min():.0f}-{df['age'].max():.0f} godina")

    # fnlwgt - veliki raspon, potvrda da je statistički ponder i da ga treba ukloniti
    report_lines.append(f"  fnlwgt: opseg {df['fnlwgt'].min():.0f}-{df['fnlwgt'].max():.0f} - vrlo širok raspon potvrđuje da je ovo statistički ponder bez direktne prediktivne vrednosti -> uklonjen u preprocesiranju")


def plot_correlation_heatmap(df, figures_dir):
    numeric_df = df[NUMERIC_COLS].copy()
    # income kao 0/1 za korelaciju sa ciljnom promenljivom
    numeric_df['income'] = (df['income'] == '>50K').astype(int)

    corr = numeric_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Korelaciona matrica numeričkih atributa")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"), dpi=120)
    plt.close()


def plot_outliers(df, figures_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flatten(), NUMERIC_COLS):
        sns.boxplot(y=df[col], ax=ax, color="skyblue")
        ax.set_title(col)
    plt.suptitle("Distribucije numeričkih atributa - detekcija outlier-a")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "numeric_outliers_boxplots.png"), dpi=120)
    plt.close()


def plot_income_distribution(df, figures_dir):
    plt.figure(figsize=(5, 4))
    sns.countplot(x='income', hue='income', data=df, palette="mako", legend=False)
    plt.title("Raspodela ciljne promenljive (income)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "income_distribution.png"), dpi=120)
    plt.close()


def plot_income_by_categorical(df, figures_dir):
    # Odnos prihoda i ključnih kategorijskih/numeričkih atributa
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.countplot(x='education', hue='income', data=df,
                  order=df['education'].value_counts().index, ax=axes[0, 0], palette="mako")
    axes[0, 0].set_title("Prihod po nivou obrazovanja")
    axes[0, 0].tick_params(axis='x', rotation=75)

    sns.countplot(x='sex', hue='income', data=df, ax=axes[0, 1], palette="mako")
    axes[0, 1].set_title("Prihod po polu")

    sns.boxplot(x='income', y='hours-per-week', hue='income', data=df, ax=axes[1, 0], palette="mako", legend=False)
    axes[1, 0].set_title("Radni sati nedeljno po prihodu")

    sns.boxplot(x='income', y='age', hue='income', data=df, ax=axes[1, 1], palette="mako", legend=False)
    axes[1, 1].set_title("Starost po prihodu")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "income_relationships.png"), dpi=120)
    plt.close()


def run_eda(train_path="dataset/raw/adult_train.csv",
            test_path="dataset/raw/adult_test.csv",
            results_dir="results/"):
    figures_dir = os.path.join(results_dir, "figures", "eda")
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Greška: Fajlovi nisu pronađeni na putanjama:\n - {train_path}\n - {test_path}")
        return

    df = load_and_clean(train_path, test_path)

    report_lines = ["=== EKSPLORATIVNA ANALIZA (EDA) - Adult Census Income ===\n"]
    report_lines.append(f"Ukupan broj redova (train + test, nakon uklanjanja tekstualnog šuma): {len(df)}\n")

    report_missing_and_anomalies(df, report_lines)

    plot_correlation_heatmap(df, figures_dir)
    plot_outliers(df, figures_dir)
    plot_income_distribution(df, figures_dir)
    plot_income_by_categorical(df, figures_dir)

    report_lines.append("\n=== SAČUVANI GRAFICI ===")
    report_lines.append(" - correlation_heatmap.png")
    report_lines.append(" - numeric_outliers_boxplots.png")
    report_lines.append(" - income_distribution.png")
    report_lines.append(" - income_relationships.png")

    with open(os.path.join(metrics_dir, "eda_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("[EDA] Analiza završena. Rezultati u 'results/metrics/eda_report.txt', grafici u 'results/figures/eda/'.")


if __name__ == "__main__":
    run_eda()
