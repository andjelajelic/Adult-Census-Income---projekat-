import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def process_dataset(df, label_encoders=None, is_train=True):
    """
    Čisti podatke i enkodira kategorijske promenljive u skladu sa novim Pandas 3+ standardima.
    """
    # 1. Čišćenje skrivenih praznih mesta (razmaka) unutar tekstualnih ćelija
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()
        
    # 2. Obrada nedostajućih vrednosti koje su označene sa '?' 
    # (Popravljeno za Copy-on-Write: izbegavamo inplace=True na pojedinačnim kolonama)
    df = df.replace('?', np.nan)
    for col in ['workclass', 'occupation', 'native-country']:
        if col in df.columns:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
            
    # 3. Uklanjanje fnlwgt atributa (statistički ponder)
    if 'fnlwgt' in df.columns:
        df = df.drop(columns=['fnlwgt'])
        
    # 4. Enkodiranje ciljne kolone (hvatamo i '>50K' i '>50K.' zbog tačke na kraju)
    if 'income' in df.columns:
        df['income'] = df['income'].apply(lambda x: 1 if '>' in str(x) else 0)
        
    # 5. Enkodiranje ostalih kategorijskih promenljivih
    new_encoders = {}
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col != 'income':
            if is_train:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                new_encoders[col] = le
            else:
                if label_encoders and col in label_encoders:
                    le = label_encoders[col]
                    # Mapiranje nepoznatih klasa ako se pojave na testu
                    df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
                    df[col] = le.transform(df[col])
                    
    return df, new_encoders

def prepare_data(train_path="dataset/raw/adult_train.csv", test_path="dataset/raw/adult_test.csv", processed_dir="dataset/processed/"):
    """
    Učitava podeljene podatke, čisti neočekivane tekstualne komentare sa dna fajla, 
    preprocesira ih i bezbedno čuva.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Greška: Fajlovi nisu pronađeni na putanjama:\n - {train_path}\n - {test_path}")
        return
        
    # Definisanje tačnih naziva kolona po specifikaciji dataset-a
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 'race', 'sex', 
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    # Učitavanje uz preskakanje loših redova
    df_train = pd.read_csv(train_path, header=None, names=column_names, on_bad_lines='skip')
    df_test = pd.read_csv(test_path, header=None, names=column_names, on_bad_lines='skip')
    
    # [REŠENJE ZA SKRIVENI TEKST]: Pretvaramo 'age' u numerički tip. 
    # Sve linije teksta (poput '|1x3 Cross validator') će postati NaN, a onda ih brišemo sa .dropna()
    df_train['age'] = pd.to_numeric(df_train['age'], errors='coerce')
    df_test['age'] = pd.to_numeric(df_test['age'], errors='coerce')
    
    df_train = df_train.dropna(subset=['age'])
    df_test = df_test.dropna(subset=['age'])
    
    # Čišćenje potencijalnih razmaka u samim nazivima kolona
    df_train.columns = df_train.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()
    
    # Pokretanje čišćenja i enkodiranja
    df_train_proc, encoders = process_dataset(df_train, is_train=True)
    df_test_proc, _ = process_dataset(df_test, label_encoders=encoders, is_train=False)
    
    # Razdvajanje ulaznih atributa (X) i ciljne promenljive (y)
    X_train = df_train_proc.drop(columns=['income'])
    y_train = df_train_proc['income']
    X_test = df_test_proc.drop(columns=['income'])
    y_test = df_test_proc['income']
    
    # Standardizacija / Skaliranje numeričkih podataka
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Čuvanje procesiranih podataka u CSV fajlove unutar dataset/processed/
    X_train_scaled.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    
    # Čuvanje enkodera i skalera kao .pkl artefakata
    with open("models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    print("[DATA PREPARATION] Uspešno završeno! Sve anomalije iz fajlova su očišćene.")

if __name__ == "__main__":
    prepare_data()