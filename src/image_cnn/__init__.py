import tensorflow as tf
from os.path import exists
from os import scandir
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


class ASLClassifier(object):
    """
    creates the model object
    """
    def __init__(self):
        self._data = {
            'train_set': [],
            'test_set': []
        }

    @property
    def train_set(self):
        return self._data['train_set']

    @property
    def test_set(self):
        return self._data['test_set']


    """
    Create the train/test datasets 
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
                horizontal_flip=True,
                vertical_flip=True,
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
    Train the model on the provided dataset
    """
    def train(self, epochs):
        pass

    """
    Identify a provided image
    """
    def ident(self, img):
        pass




    """
    Displays the statistics of the trained model (accuracy, confusion_matrix, classification report)
    """
    def performance_report(self):
        pass

