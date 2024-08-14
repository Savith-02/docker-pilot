import os
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)

# Load TFLite model
interpreter = tflite.Interpreter(model_path="./model/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        # Respond to preflight request
        return add_cors_headers(make_response('', 204))

    data = request.get_json(force=True)
    input_data = np.array(data, dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.round(output_data).astype(int)
    # Change to 7 if 0-7 is desired range
    output_data = np.clip(output_data, None, 7)

    response = make_response(jsonify(output_data.tolist()))
    return add_cors_headers(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 3000)))
