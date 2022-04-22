import tensorflow as tf
from os.path import exists
from os import scandir
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ASLClassifier(object):
    """
    creates the model object # TODO: Actually comment this
    """
    def __init__(self):
        self._data = {
            'train_set': [],
            'test_set': []
        }
        self._model = None
        self._input_shape = None
        self._kernel_size = (5, 5)

    # region Properties

    @property
    def train_set(self):
        return self._data['train_set']

    @property
    def test_set(self):
        return self._data['test_set']

    @property
    def model(self):
        return self._model
    # endregion

    """
    Create the train/test datasets (Relies heavily on Tensorflow's ImageDataGenerator: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
    """
    def load_dataset(self, path_to_train, split=0.3, img_shape=(200, 200, 3), batch_size=32, image_gen=None):
        if not exists(path_to_train):
            raise FileNotFoundError(f"'{path_to_train}' could not be located!")

        # Setting up the image preprocessor
        if image_gen is not None and isinstance(image_gen, ImageDataGenerator):
            img_gen = image_gen
        else:
            img_gen = ImageDataGenerator(
                rescale=1.0/255.0,
                horizontal_flip=False,
                vertical_flip=False,
                validation_split=split
            )


        # Getting the training set
        self._data['train_set'] = img_gen.flow_from_directory(
            path_to_train,
            target_size=img_shape[:2],
            color_mode='rgb',
            batch_size=batch_size,
            subset='training',
            class_mode='categorical',
            shuffle=True
        )
        # Getting the testing set
        self._data['test_set'] = img_gen.flow_from_directory(
            path_to_train,
            target_size=img_shape[:2],
            color_mode='rgb',
            batch_size=batch_size,
            subset='validation',
            class_mode='categorical',
            shuffle=False
        )

        self._input_shape = img_shape


    def display_25_img(self, img_set="train_set"):
        if img_set != "train_set" and img_set != "test_set":
            raise ValueError("<img_type> must either be 'train_set' or 'test_set'")
        plt.figure(figsize=(10, 10))
        images, labels = self._data[img_set].next()

        for i in range(25):
            ax = plt.subplot(5, 5, i+1)
            plt.imshow(images[i])
            plt.axis('off')

    """
    Creates and Compiles the model
    """
    def compile_model(self, seq_model=None, model_layers=None, model_optimizer='Adam'):
        if self._input_shape is None:
            raise AttributeError("No dataset has been loaded")

        if seq_model is not None and layers is not None:
            raise ValueError("<seq_model: keras.Sequential> and <layers: list> are mutually exclusive! Provide one or the other or neither.")

        if seq_model is not None and isinstance(seq_model, Sequential):
            self._model = seq_model
        elif layers is not None and isinstance(layers, list):
            self._model = Sequential(model_layers)

        else:
            # Without any alternative model being specified, default
            self._model = Sequential([
                layers.Conv2D(input_shape=self._input_shape, filters=16, kernel_size=self._kernel_size, activation='relu', padding='same'),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Dropout(0.4),
                # block 1
                layers.Conv2D(filters=32, kernel_size=self._kernel_size, activation='relu', padding='same'),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Dropout(0.4),
                # block 2
                layers.Conv2D(filters=64, kernel_size=self._kernel_size, activation='relu', padding='same'),
                layers.MaxPool2D(pool_size=(2, 2)),
                layers.Dropout(0.4),
                # block 3
                layers.Conv2D(filters=64, kernel_size=self._kernel_size, activation='relu', padding='same'),
                layers.MaxPool2D(pool_size=(2,2)),
                layers.Dropout(0.4),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(29, activation='sigmoid')
            ])
        self._model.compile(loss='categorical_crossentropy', optimizer=model_optimizer, metrics=['accuracy'])

    """
    Train the model on the provided dataset
    """

    def train(self, epochs=20, early_stop = None):  # TODO: Add option for early stopping should be added
        if self._model is None:
            raise AttributeError("No model has been compiled! Call '<ASLClassifier>.compile_model()'")
        return self._model.fit(self._data['train_set'], epochs=epochs, validation_data=self._data['test_set'])

    """
    Identify a provided image
    """
    def ident(self, img):
        pass


    """
    Displays the statistics of the trained model (accuracy, confusion_matrix, classification report)
    """
    def performance_report(self, prediction):

        print('Classification report: \n', classification_report(prediction, self._data['test_set'].classes))
        # print('Confusion_matrix: \n', confusion_matrix(prediction, self._data['test_set'].classes))
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(confusion_matrix(prediction, self._data['test_set'].classes), annot=True)


