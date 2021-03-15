import idx2numpy
import numpy
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM, \
    BatchNormalization
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tools import emnist_labels, emnist_path, dir
from tst2 import *


def CreateModel_v1():
    model = Sequential()
    model.add(
        Convolution2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(33, activation="softmax"))
    # model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
    #               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    return model

def CreateModel_v2():
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(emnist_labels), activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
    return model




# def prepareData(k=5):
#     X_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-images-idx3-ubyte')
#     y_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-labels-idx1-ubyte')
#
#     X_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-images-idx3-ubyte')
#     y_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-labels-idx1-ubyte')
#
#     X_train = numpy.reshape(X_train, (X_train.shape[0], 28, 28, 1))
#     X_test = numpy.reshape(X_test, (X_test.shape[0], 28, 28, 1))
#
#     print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, len(emnist_labels))
#
#     X_train = X_train[:X_train.shape[0] // k]
#     y_train = y_train[:y_train.shape[0] // k]
#     X_test = X_test[:X_test.shape[0] // k]
#     y_test = y_test[:y_test.shape[0] // k]
#
#     # Normalize
#     X_train = X_train.astype(numpy.float32)
#     X_train /= 255.0
#     X_test = X_test.astype(numpy.float32)
#     X_test /= 255.0
#
#     x_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
#     y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))
#     return X_train, x_train_cat, X_test, y_test_cat

def prepareData2():
    ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
    TRAINING_DIR = "C:\\Users\\farad\\Desktop\\OneDrive\\Dev\\cnn\\Cyrillic_cln_64_r"
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                        batch_size=40,
                                                        class_mode='binary',
                                                        target_size=(64, 64))

    VALIDATION_DIR = "C:\\Users\\farad\\Desktop\\OneDrive\\Dev\\cnn\\Cyrillic_cln_64_r"
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                                  batch_size=40,
                                                                  class_mode='binary',
                                                                  target_size=(64, 64))
    return train_generator, validation_generator


def lernNN(model, fileName: str):
    X_train, x_train_cat, X_test, y_test_cat = prepareData()

    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1,
                                                                factor=0.5,
                                                                min_lr=0.00001)

    # train_generator, validation_generator = prepareData2()
    #
    # history = model.fit_generator(train_generator,
    #                        epochs=300,
    #                        verbose=1,
    #                        )

    model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction],
              batch_size=64, epochs=200)

    model.save("models/" + fileName + ".h5")

