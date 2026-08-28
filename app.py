# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import re

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')

    if not message.strip():
        return jsonify({'error': 'Message is empty'}), 400

    cleaned = clean_text(message)
    prediction = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]

    result = {
        'prediction': 'Spam' if prediction == 1 else 'Not Spam',
        'confidence': float(proba[prediction]) * 100,
        'spam_probability': float(proba[1]) * 100,
        'ham_probability': float(proba[0]) * 100
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
