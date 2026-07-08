import os
import pickle
import pandas as pd

# Ova skripta se nalazi u src/, a models/ je u korenu projekta (isti pristup kao predict.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Redosled i nazivi atributa koje model očekuje (isti redosled kao X_train.csv,
# tj. originalne kolone bez 'fnlwgt' i 'income')
RAW_FEATURE_COLUMNS = [
    'age', 'workclass', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]

CATEGORICAL_COLUMNS = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']

NUMERIC_COLUMNS = [c for c in RAW_FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

_artifacts_cache = {}


def load_artifacts(models_dir=MODELS_DIR):
    """Učitava model, enkodere i skaler jednom i drži ih u memoriji (cache)."""
    if not _artifacts_cache:
        with open(os.path.join(models_dir, "best_rf_model.pkl"), "rb") as f:
            _artifacts_cache["model"] = pickle.load(f)
        with open(os.path.join(models_dir, "encoders.pkl"), "rb") as f:
            _artifacts_cache["encoders"] = pickle.load(f)
        with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
            _artifacts_cache["scaler"] = pickle.load(f)
    return _artifacts_cache


def preprocess_raw(input_dict, models_dir=MODELS_DIR):
    """
    Prima rečnik sa SIROVIM, ljudski čitljivim vrednostima za jednu osobu
    (npr. workclass='Private', education='Bachelors', age=39, ...) i vraća
    DataFrame enkodiran i skaliran na isti način kao u data_preparation.py,
    spreman za model.predict().
    """
    artifacts = load_artifacts(models_dir)
    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]

    missing = [c for c in RAW_FEATURE_COLUMNS if c not in input_dict]
    if missing:
        raise ValueError(f"Nedostaju obavezna polja: {missing}")

    df = pd.DataFrame([input_dict])[RAW_FEATURE_COLUMNS].copy()

    # Čišćenje razmaka u tekstualnim vrednostima
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).str.strip()

    # Enkodiranje kategorijskih atributa - isti pristup kao u data_preparation.py:
    # nepoznate vrednosti se mapiraju na prvu poznatu klasu umesto da izazovu grešku
    for col in CATEGORICAL_COLUMNS:
        le = encoders[col]
        df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    # Numerički atributi
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if df[NUMERIC_COLUMNS].isna().any().any():
        raise ValueError("Jedan ili više numeričkih atributa nije moguće konvertovati u broj.")

    # Skaliranje - scaler je treniran na kolonama u istom redosledu (RAW_FEATURE_COLUMNS)
    df_scaled = pd.DataFrame(
        scaler.transform(df[RAW_FEATURE_COLUMNS]),
        columns=RAW_FEATURE_COLUMNS
    )
    return df_scaled


def predict_single(input_dict, models_dir=MODELS_DIR):
    """Vraća predikciju i verovatnoću za jednu osobu na osnovu sirovih podataka."""
    artifacts = load_artifacts(models_dir)
    model = artifacts["model"]

    X = preprocess_raw(input_dict, models_dir)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return {
        "prediction": ">50K" if pred == 1 else "<=50K",
        "probability_over_50k": round(prob, 4)
    }
