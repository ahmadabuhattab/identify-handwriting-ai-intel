import unittest
from src.data_preprocessing import load_and_preprocess_data
from src.model import create_advanced_model

class TestHandwritingRecognition(unittest.TestCase):
    def test_data_loading(self):
        (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
        self.assertEqual(x_train.shape[1:], (28, 28, 1))
        self.assertEqual(y_train.shape[1], 10)
    
    def test_model_creation(self):
        model = create_advanced_model()
        self.assertEqual(model.input_shape, (None, 28, 28, 1))
        self.assertEqual(model.output_shape, (None, 10))

if __name__ == "__main__":
    unittest.main()
