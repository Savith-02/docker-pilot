import os
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all domains and routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Load TFLite model
interpreter = tflite.Interpreter(model_path="./model/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Add this line to debug the input data structure
    print("Received data:", data)

    # Parse the input string into a list of numbers
    input_list = json.loads(data['input'])

    # Convert the list to a numpy array
    input_data = np.array(input_list, dtype=np.float32)

    print("Input data:", input_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.round(output_data).astype(int)
    output_data = np.clip(output_data, None, 8)
    print("Output data:", output_data)
    return jsonify({'prediction': output_data.tolist()})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
