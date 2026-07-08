from flask import Flask, request, jsonify
from inference_utils import predict_single, RAW_FEATURE_COLUMNS, CATEGORICAL_COLUMNS

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    """Osnovne informacije o API-ju - koristno za brzu proveru da li server radi."""
    return jsonify({
        "status": "ok",
        "message": "Adult Census Income - Prediction API",
        "endpoint": "/predict (POST)",
        "expected_fields": RAW_FEATURE_COLUMNS,
        "categorical_fields": CATEGORICAL_COLUMNS,
        "example_payload": {
            "age": 39, "workclass": "State-gov", "education": "Bachelors",
            "education-num": 13, "marital-status": "Never-married",
            "occupation": "Adm-clerical", "relationship": "Not-in-family",
            "race": "White", "sex": "Male", "capital-gain": 2174,
            "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prima JSON sa sirovim (ljudski čitljivim) podacima o jednoj osobi i vraća
    predikciju da li je prihod >50K ili <=50K, uz verovatnoću.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Telo zahteva mora biti validan JSON."}), 400

    try:
        result = predict_single(data)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Neočekivana greška: {e}"}), 500


if __name__ == "__main__":
    print("[API] Pokrećem Flask server na http://localhost:5000 ...")
    print("[API] Prvo pokreni data_preparation.py i train.py ako to još nisi uradila.")
    app.run(debug=True, port=5000)
