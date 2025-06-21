from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return "Iris Classifier is up!"

# JSON API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

# HTML form route
@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        try:
            f1 = float(request.form["f1"])
            f2 = float(request.form["f2"])
            f3 = float(request.form["f3"])
            f4 = float(request.form["f4"])
            features = np.array([f1, f2, f3, f4]).reshape(1, -1)
            class_names = ["Setosa", "Versicolor", "Virginica"]
            prediction = model.predict(features)[0]
            flower_name = class_names[prediction]
            return render_template_string(html_form, prediction=flower_name)
        except:
            return render_template_string(html_form, prediction="Invalid input")
    return render_template_string(html_form, prediction=None)

# Embedded HTML
html_form = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Iris Flower Predictor</title>
    <style>
        body {
            background: linear-gradient(120deg, #84fab0, #8fd3f4);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            background-color: white;
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
            width: 350px;
            text-align: center;
        }
        h2 {
            color: #333;
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 10px;
            text-align: left;
            font-weight: bold;
            color: #555;
        }
        input[type="text"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 6px;
            border: 1px solid #ccc;
            box-sizing: border-box;
            font-size: 14px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            font-size: 18px;
            color: #333;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Iris Prediction Form 🌸</h2>
        <form method="POST">
            <label>Sepal Length</label>
            <input type="text" name="f1" placeholder="e.g., 5.1">

            <label>Sepal Width</label>
            <input type="text" name="f2" placeholder="e.g., 3.5">

            <label>Petal Length</label>
            <input type="text" name="f3" placeholder="e.g., 1.4">

            <label>Petal Width</label>
            <input type="text" name="f4" placeholder="e.g., 0.2">

            <button type="submit">🔍 Predict</button>
        </form>
        {% if prediction is not none %}
            <div class="result">
                🌼 Prediction: {{ prediction }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5060, debug=True)
