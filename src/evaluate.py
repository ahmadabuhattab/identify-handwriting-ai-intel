import tensorflow as tf
from src.data_preprocessing import load_and_preprocess_data

def evaluate_model():
    (_, _), (x_test, y_test) = load_and_preprocess_data()
    model = tf.keras.models.load_model('handwriting_model_advanced.h5')
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)

if __name__ == "__main__":
    evaluate_model()
