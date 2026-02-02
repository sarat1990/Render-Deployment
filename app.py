from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model once at startup
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Expecting a list: [sepal_length, sepal_width, petal_length, petal_width]
        prediction = model.predict([np.array(data['features'])])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
