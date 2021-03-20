import cv2
import numpy
import os
from tensorflow import keras
from letters import alph

train_dir = "C:\\Users\\Faraday\\Desktop\\OneDrive\\Dev\\cnn\\64_train\\"
test_dir = "C:\\Users\\Faraday\\Desktop\\OneDrive\\Dev\\cnn\\64_test\\"

emnist_path = "C:\\wrk\\cnn\\emnist\\"


def prepareData(size=28):
    k=1

    #load train data
    X_train = []
    y_train = []
    for c in os.listdir(train_dir):
        for i in os.listdir(train_dir + c):
            img = cv2.imread(train_dir + c + "\\" + i, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            X_train.append(img)
            y_train.append(int(c))
    X_train = numpy.array(X_train)
    y_train = numpy.array(y_train)

    #load test data
    X_test = []
    y_test = []
    for c in os.listdir(test_dir):
        for i in os.listdir(test_dir + c):
            img = cv2.imread(test_dir + c + "\\" + i, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            X_test.append(img)
            y_test.append(int(c))
    X_test = numpy.array(X_test)
    y_test = numpy.array(y_test)

    X_train = numpy.reshape(X_train, (X_train.shape[0], size, size, 1))
    X_test = numpy.reshape(X_test, (X_test.shape[0], size, size, 1))

    X_train = X_train[:X_train.shape[0] // k]
    y_train = y_train[:y_train.shape[0] // k]
    X_test = X_test[:X_test.shape[0] // k]
    y_test = y_test[:y_test.shape[0] // k]

    # Normalize
    X_train = X_train.astype(numpy.float32)
    X_train =1- X_train/255.0
    X_test = X_test.astype(numpy.float32)
    X_test = 1-X_test/255.0

    x_train_cat = keras.utils.to_categorical(y_train, len(alph))
    y_test_cat = keras.utils.to_categorical(y_test, len(alph))
    # x_train_cat = y_train
    # y_test_cat = y_test
    return X_train, x_train_cat, X_test, y_test_cat


def prepareData2(size=28):
    ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
    TRAINING_DIR = train_dir
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                        batch_size=40,
                                                        class_mode='binary',
                                                        target_size=(size, size))

    VALIDATION_DIR = test_dir
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                                  batch_size=40,
                                                                  class_mode='binary',
                                                                  target_size=(size, size))
    return train_generator, validation_generator