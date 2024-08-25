from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('handwriting_model_advanced.h5')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['file'].read()
    prediction = model.predict(image)
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
