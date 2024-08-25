import kerastuner as kt
from src.data_preprocessing import load_and_preprocess_data
from src.model import create_advanced_model

def model_builder(hp):
    model = create_advanced_model()
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def tune_hyperparameters():
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    best_hps = tuner.get_best_hyperparameters()[0]
    print(f"Best hyperparameters: {best_hps.values}")

if __name__ == "__main__":
    tune_hyperparameters()
