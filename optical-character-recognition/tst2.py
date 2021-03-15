import cv2
import idx2numpy
import numpy
import os
from model import *


train_dir = "C:\\Users\\Faraday\\Desktop\\OneDrive\\Dev\\cnn\\64_train\\"
test_dir = "C:\\Users\\Faraday\\Desktop\\OneDrive\\Dev\\cnn\\64_test\\"

emnist_path = "C:\\wrk\\cnn\\emnist\\"


def prepareData(size = 28):
    X_train = []
    y_train = []
    for c in os.listdir(train_dir):
        for i in os.listdir(train_dir + c):
            img = cv2.imread(train_dir + c + "\\" + i, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
            X_train.append(img)
            y_train.append(int(c))

    X_train = numpy.array(X_train)
    y_train = numpy.array(y_train)
    print("Train\n" + str(X_train.shape))
    print(y_train.shape, "\n")

    X_test = []
    y_test = []
    for c in os.listdir(test_dir):
        for i in os.listdir(test_dir + c):
            img = cv2.imread(test_dir + c + "\\" + i, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
            X_test.append(img)
            y_test.append(int(c))
    X_test = numpy.array(X_test)
    y_test = numpy.array(y_test)
    print("Test\n" + str(X_test.shape))
    print(y_test.shape, "\n")


    X_train = numpy.reshape(X_train, (X_train.shape[0], size, size, 1))
    X_test = numpy.reshape(X_test, (X_test.shape[0], size, size, 1))

    print(X_train.shape)
    print(X_test.shape, "\n")

    # Normalize
    X_train = X_train.astype(numpy.float32)
    X_train /= 255.0
    X_test = X_test.astype(numpy.float32)
    X_test /= 255.0

    # x_train_cat = keras.utils.to_categorical(y_train, 33)
    # y_test_cat = keras.utils.to_categorical(y_test, 33)
    return X_train, x_train_cat, X_test, y_test_cat

if __name__ == '__main__':
    k=20

    X_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-images-idx3-ubyte')
    y_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-labels-idx1-ubyte')

    X_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-images-idx3-ubyte')
    y_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-labels-idx1-ubyte')


    X_train = numpy.reshape(X_train, (X_train.shape[0], 28, 28, 1))
    X_test = numpy.reshape(X_test, (X_test.shape[0], 28, 28, 1))

    X_train = X_train[:X_train.shape[0] // k]
    y_train = y_train[:y_train.shape[0] // k]
    X_test = X_test[:X_test.shape[0] // k]
    y_test = y_test[:y_test.shape[0] // k]

    # Normalize
    X_train = X_train.astype(numpy.float32)
    X_train /= 255.0
    X_test = X_test.astype(numpy.float32)
    X_test /= 255.0

    x_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
    y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))

    print(y_train.shape)

    #prepareData(28)

    # cv2.imshow("0",X_test[5])
    # cv2.waitKey(0)
