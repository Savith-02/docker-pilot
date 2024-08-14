import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Enable CORS for all domains and routes
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

# Load TFLite model
interpreter = tflite.Interpreter(model_path="./model/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route('/predict', methods=['POST', 'OPTIONS'])
@cross_origin()
def predict():
    data = request.get_json(force=True)
    # Add this line to debug the input data structure
    print("Received data:", data)
    input_data = np.array(data['input'], dtype=np.float32)
    print("Input data:", input_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.round(output_data).astype(int)
    output_data = np.clip(output_data, None, 8)
    print("Output data:", output_data)
    return jsonify({'prediction': output_data.tolist()})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 3000)))
