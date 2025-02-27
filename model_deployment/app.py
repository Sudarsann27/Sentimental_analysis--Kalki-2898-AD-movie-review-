from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer with error handling
try:
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model, vectorizer = None, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return render_template('index.html', error="Model loading failed. Please check the server.")

    review = request.form.get('review', '').strip()

    if not review:
        return render_template('index.html', error="Please enter a review before submitting.")

    try:
        data = vectorizer.transform([review])
        prediction = model.predict(data)
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {e}")

    return render_template('index.html', review=review, sentiment=sentiment)

@app.route('/predict_json', methods=['POST'])
def predict_json():
    if not model or not vectorizer:
        return jsonify({"error": "Model loading failed. Please check the server."}), 500

    try:
        data = request.get_json()
        review = data.get("review", "").strip()

        if not review:
            return jsonify({"error": "No review provided"}), 400

        vectorized_data = vectorizer.transform([review])
        prediction = model.predict(vectorized_data)
        sentiment = 'Negative' if prediction[0] == 1 else 'positive'

        return jsonify({"review": review, "sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)