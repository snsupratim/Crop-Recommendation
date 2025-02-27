from flask import Flask, request, render_template
import numpy as np
import pickle
import os
from flask import jsonify
from flask_cors import CORS

# Load the trained model
with open("crop_recommendation.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load label encoder (to decode crop names)
with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)



app = Flask(__name__)
CORS(app)  # Allow all origins


@app.route("/")
def home():
    return render_template("index.html")  # Load HTML form



@app.route("/predict", methods=["POST"])
def predict():
    try:
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_data)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

        image_filename = f"static/crop-img/{predicted_crop.lower()}.png"
        if not os.path.exists(image_filename):
            image_filename = "static/crop-img/default.png"

        return jsonify({"prediction": predicted_crop, "image": image_filename})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
