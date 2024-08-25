from src.data_augmentation import augment_data
from src.model import create_advanced_model
from src.hyperparameter_tuning import tune_hyperparameters

def train_model():
    datagen, x_train, y_train, x_test, y_test = augment_data()
    tune_hyperparameters()
    model = create_advanced_model()
    model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=20, validation_data=(x_test, y_test))
    model.save('handwriting_model_advanced.h5')

if __name__ == "__main__":
    train_model()
