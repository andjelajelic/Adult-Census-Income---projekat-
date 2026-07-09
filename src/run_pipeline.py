import os
import sys
import subprocess

STEPS = [
    ("Eksplorativna analiza podataka (EDA)", "eda.py"),
    ("Priprema, čišćenje i enkodiranje podataka", "data_preparation.py"),
    ("Treniranje modela (Logistic Regression + Random Forest)", "train.py"),
    ("Evaluacija i poređenje modela", "evaluate.py"),
    ("Odabir najznačajnijih atributa (RFE)", "feature_selection.py"),
    ("Primer predikcije nad test skupom", "predict.py"),
]


def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))

    for i, (description, script) in enumerate(STEPS, start=1):
        print(f"\n{'=' * 65}")
        print(f"[{i}/{len(STEPS)}] {description}  ({script})")
        print("=" * 65)

        result = subprocess.run([sys.executable, os.path.join(src_dir, script)])

        if result.returncode != 0:
            print(f"\n[PIPELINE] Korak '{script}' je prekinut greškom. Zaustavljam pipeline.")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("[PIPELINE] Svi koraci su uspešno završeni!")
    print("Rezultati: results/   |   Sačuvani modeli: models/")
    print("Za API i UI pokreni odvojeno: python src/app.py  i  streamlit run src/streamlit_app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
