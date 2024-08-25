from src.data_preprocessing import load_and_preprocess_data, create_data_augmentor

def augment_data():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    datagen = create_data_augmentor()
    datagen.fit(x_train)
    return datagen, x_train, y_train, x_test, y_test
