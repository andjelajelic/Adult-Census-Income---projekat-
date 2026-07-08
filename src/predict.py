import os
import pickle
import pandas as pd

def predict_income(input_data, models_dir="models/"):
    """
    input_data: Može biti rečnik (za jednu osobu) ili pandas DataFrame (za više ljudi).
    Imaj na umu da ulazni podaci moraju biti prethodno enkodirani i skalirani kroz isti skaler.
    """
    # 1. POPRAVKA PUTANJE: Nalazimo koren projekta u odnosu na ovu skriptu
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, models_dir, "best_rf_model.pkl")
    
    if not os.path.exists(model_path):
        return "Model nije pronađen. Prvo pokrenite train.py."
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    else:
        df_input = input_data
        
    predictions = model.predict(df_input)
    return [" >50K" if p == 1 else " <=50K" for p in predictions]


# --- OVAJ DEO DODAJEŠ NA SAM KRAJ FAJLA DA BI MOGAO DA POKRENEŠ SKRIPTU ---
# --- OVAJ DEO ZAMENI NA KRAJU FAJLA ---
if __name__ == "__main__":
    print("[PREDICT] Pokrećem testno predviđanje pomoću X_test.csv...")
    
    # Nalazimo koren projekta
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    x_test_putanja = os.path.join(BASE_DIR, "dataset", "processed", "X_test.csv")
    
    if os.path.exists(x_test_putanja):
        # Učitavamo samo prvi red iz već procesuiranog X_test fajla
        # Koristimo .iloc[[0]] da zadržimo DataFrame strukturu (sa nazivima kolona)
        test_podaci = pd.read_csv(x_test_putanja).iloc[[0]] 
        
        print("\nUspesno učitan testni red. Kolone u fajlu:")
        print(test_podaci.columns.tolist()[:5], "... i ostale.") # Prikaz prvih 5 kolona čisto radi provere
        
        # Pokrećemo predviđanje
        rezultat = predict_income(test_podaci)
        print(f"\n[REZULTAT] Predviđanje za ovaj red je: {rezultat}")
    else:
        print(f"\n[GREŠKA] Ne mogu da pronađem X_test.csv na putanji: {x_test_putanja}")
        print("Proveri da li si pokrenuo preprocess.py skriptu pre ovoga.")