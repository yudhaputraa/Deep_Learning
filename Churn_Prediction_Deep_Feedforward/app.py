from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('best_model.h5')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    # Convert data into numpy array
    input_data = np.array([data['features']])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Convert prediction to a list and return as JSON
    output = prediction[0].tolist()
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)
