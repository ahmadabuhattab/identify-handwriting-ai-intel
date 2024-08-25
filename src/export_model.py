import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot

def export_optimized_model():
    model = load_model('handwriting_model_advanced.h5')
    model = tfmot.quantization.keras.quantize_model(model)
    model.save('handwriting_model_optimized.tflite')

if __name__ == "__main__":
    export_optimized_model()
