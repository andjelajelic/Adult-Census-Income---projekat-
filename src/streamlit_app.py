import os
import pickle
import requests
import streamlit as st

API_URL = "http://localhost:5000/predict"
# Ova skripta se nalazi u src/, a models/ je u korenu projekta (isti pristup kao predict.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Adult Census Income - Predikcija", page_icon="💰")


@st.cache_resource
def load_encoders():
    """
    Enkoderi se učitavaju SAMO da bismo popunili padajuće liste (dropdown) sa
    validnim kategorijama koje model poznaje. Sama predikcija ide preko Flask API-ja.
    """
    with open(os.path.join(BASE_DIR, "models", "encoders.pkl"), "rb") as f:
        return pickle.load(f)


st.title("💰 Adult Census Income - Predikcija prihoda")
st.write(
    "Unesite demografske i profesionalne podatke o osobi da biste predvideli "
    "da li joj je godišnji prihod veći od 50.000 dolara."
)

try:
    encoders = load_encoders()
except FileNotFoundError:
    st.error(
        "Nisu pronađeni sačuvani enkoderi u 'models/encoders.pkl'. "
        "Prvo pokreni 'data_preparation.py' i 'train.py'."
    )
    st.stop()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Starost", min_value=17, max_value=90, value=35)
        workclass = st.selectbox("Tip zaposlenja", encoders["workclass"].classes_)
        education = st.selectbox("Nivo obrazovanja", encoders["education"].classes_)
        education_num = st.slider("Broj godina školovanja (education-num)", 1, 16, 10)
        marital_status = st.selectbox("Bračni status", encoders["marital-status"].classes_)
        occupation = st.selectbox("Zanimanje", encoders["occupation"].classes_)
        relationship = st.selectbox("Porodična uloga", encoders["relationship"].classes_)

    with col2:
        race = st.selectbox("Rasa", encoders["race"].classes_)
        sex = st.selectbox("Pol", encoders["sex"].classes_)
        capital_gain = st.number_input("Kapitalna dobit ($)", min_value=0, value=0, step=100)
        capital_loss = st.number_input("Kapitalni gubitak ($)", min_value=0, value=0, step=100)
        hours_per_week = st.slider("Radni sati nedeljno", 1, 99, 40)
        native_country = st.selectbox("Država rođenja", encoders["native-country"].classes_)

    submitted = st.form_submit_button("Predvidi prihod", use_container_width=True)

if submitted:
    payload = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "education-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        label = result["prediction"]
        prob = result["probability_over_50k"]

        st.divider()
        if label == ">50K":
            st.success(f"### Predviđanje: {label}")
        else:
            st.info(f"### Predviđanje: {label}")
        st.progress(prob, text=f"Verovatnoća da prihod bude >50K: {prob:.1%}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Ne mogu da se povežem sa API-jem na http://localhost:5000. "
            "Proveri da li je 'app.py' pokrenut u drugom terminalu (python app.py)."
        )
    except requests.exceptions.HTTPError:
        st.error(f"API je vratio grešku: {response.json().get('error', 'nepoznata greška')}")
    except Exception as e:
        st.error(f"Neočekivana greška: {e}")

st.divider()
st.caption(
    "Napomena: aplikacija šalje podatke ka lokalnom Flask API-ju (app.py), "
    "koji radi predikciju pomoću istreniranog Random Forest modela."
)
