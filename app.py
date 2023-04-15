from flask import Flask, jsonify, request
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the LSTM model
model = load_model('lstm_model.h5') 

# Define the Flask app
app = Flask(__name__)

# Define the API endpoints
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()

    # Preprocess the input data
    input_data = preprocess_data(data)

    # Use the LSTM model to make a prediction
    prediction = model.predict(input_data)

    # Postprocess the prediction
    output_data = postprocess_data(prediction)

    # Return the prediction as a JSON response
    return jsonify(output_data)

# Define the preprocess_data function
def preprocess_data(data):
    # Convert the text to a sequence of integers
    seq = tokenizer.texts_to_sequences([data['text']])

    # Pad the sequence to a fixed length
    padded_seq = pad_sequences(seq, maxlen=500)

    # Return the preprocessed data
    return padded_seq

# Define the postprocess_data function
def postprocess_data(prediction):
    # Convert the prediction to a label
    label = np.argmax(prediction)

    # Return the label as the output data
    return {'label': label}

# Run the Flask app
if __name__ == '__main__':
    app.run()