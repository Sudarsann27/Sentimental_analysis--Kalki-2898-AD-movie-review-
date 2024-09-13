from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your logistic regression model
model = joblib.load('sentiment_model.pkl') 
vectorizer = joblib.load('vectorizer.pkl') 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        data = vectorizer.transform([review])
        prediction = model.predict(data)
        
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)