# Adult Census Income ML

Projekat iz predmeta **SAUSAU** koji primenjuje mašinsko učenje na predviđanje kategorije godišnjeg prihoda pojedinca na osnovu popisnih podataka SAD-a.

## Problem

Adult Census Income je klasičan skup podataka iz popisa stanovništva SAD-a koji sadrži demografske, obrazovne i profesionalne karakteristike pojedinaca. Cilj je binarna klasifikacija: da li godišnji prihod osobe prelazi **50.000 dolara** (`>50K`) ili ne (`≤50K`), na osnovu atributa poput obrazovanja, zanimanja, bračnog statusa i broja radnih sati nedeljno.

Cilj projekta je izgradnja klasifikacionog ML modela koji na osnovu ovih atributa predviđa **kategoriju prihoda** (`income`), uz identifikaciju atributa koji najviše doprinose predikciji.

## Skup podataka

Podaci su preuzeti iz UCI Adult Census Income dataseta (`dataset/raw/adult_train.csv`, `dataset/raw/adult_test.csv`) i sadrže sledeće atribute:

| Atribut | Opis |
|---|---|
| `age` | Starost osobe u godinama |
| `workclass` | Tip zaposlenja (privatni sektor, državna služba, samozaposlen...) |
| `fnlwgt` | Statistički ponder (uklonjen u preprocesiranju — ne nosi prediktivnu informaciju) |
| `education`, `education-num` | Nivo obrazovanja (tekstualno i numerički) |
| `marital-status` | Bračni status |
| `occupation` | Zanimanje |
| `relationship` | Porodična uloga u domaćinstvu |
| `race`, `sex` | Rasa i pol |
| `capital-gain`, `capital-loss` | Kapitalna dobit / gubitak |
| `hours-per-week` | Broj radnih sati nedeljno |
| `native-country` | Država rođenja |
| `income` | **Ciljna promenljiva** — `≤50K` ili `>50K` |

Dataset sadrži **48 842 instance** (32 561 trening + 16 281 test), nakon uklanjanja tekstualnog šuma iz sirovih fajlova.

## Struktura projekta

```
adult-census-income-ml/
├── dataset/
│   ├── raw/                     # Sirovi podaci (adult_train.csv, adult_test.csv)
│   └── processed/                # Enkodirani i skalirani podaci (generiše se automatski)
├── models/                       # Sačuvani modeli, enkoderi i skaler
├── results/
│   ├── figures/
│   │   ├── eda/                  # EDA grafikoni (korelacije, outlieri, distribucije)
│   │   └── ...                   # Grafikoni poređenja modela i feature importance
│   └── metrics/                  # Tekstualni i CSV izveštaji
└── src/
    ├── eda.py                    # Eksplorativna analiza podataka
    ├── data_preparation.py       # Čišćenje, enkodiranje i skaliranje podataka
    ├── train.py                  # Treniranje i podešavanje hiperparametara
    ├── evaluate.py                # Evaluacija i poređenje modela
    ├── feature_selection.py      # Odabir najznačajnijih atributa (RFE - wrapper metoda)
    ├── predict.py                  # Primer predikcije nad test skupom
    ├── inference_utils.py        # Deljena logika za obradu sirovih ulaznih podataka
    ├── app.py                      # Flask REST API za predikciju
    ├── streamlit_app.py           # Web UI za predikciju (koristi Flask API)
    └── run_pipeline.py             # Pokretanje celog pipeline-a jednom komandom
```

## Pokretanje

### Instalacija zavisnosti

```bash
pip install -r requirements.txt
```

### Pokretanje celog pipeline-a

```bash
python src/run_pipeline.py
```

Ovo izvršava sledeće korake redom:
1. Eksplorativna analiza podataka (EDA)
2. Čišćenje, enkodiranje i skaliranje podataka
3. Treniranje modela (Logistic Regression + Random Forest, uz `GridSearchCV` za hiperparametre)
4. Evaluacija i poređenje modela na test skupu
5. Odabir najznačajnijih atributa (RFE — wrapper metoda)
6. Primer predikcije nad test skupom

Svaki korak zavisi od fajlova koje je prethodni sačuvao na disk (u `dataset/processed/`, `models/`, `results/`); pipeline se zaustavlja ako neki korak ne uspe.

### Pokretanje web UI-ja i API-ja

```bash
# Terminal 1 — API
python src/app.py
```
API radi na `http://localhost:5000` (`GET /` za status, `POST /predict` za predikciju).

```bash
# Terminal 2 — UI
streamlit run src/streamlit_app.py
```
UI radi na `http://localhost:8501` i poziva API iz Terminala 1, koji mora ostati pokrenut.

## Rezultati

- **Random Forest** (optimizovan pomoću `GridSearchCV`, `cv=3`, `scoring='f1'`) nadmašuje **Logističku regresiju** na svim metrikama: F1-skor **0,668** naspram 0,547, ROC-AUC **0,902** naspram 0,852.
- Najznačajniji atributi (RFE — wrapper metoda): `age`, `education-num`, `occupation`, `relationship`, `capital-gain`, `hours-per-week`.
- Detaljna diskusija rezultata nalazi se u `Adult_Census_Income_Dokumentacija.pdf`.
