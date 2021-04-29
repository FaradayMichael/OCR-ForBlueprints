import idx2numpy
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM, \
    BatchNormalization
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from letters import alph
from data import prepareData, prepareData2
import gc




def createModel_v1(size=28):
    model = keras.Sequential()
    model.add(
        Convolution2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(size, size, 1), activation='relu'))
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
    model.add(Dense(len(alph), activation=tensorflow.nn.softmax))
    # model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    return model

def createModel_v2(size=28):
    # model = Sequential()
    # model.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(len(alph), activation='softmax'))

    # model = Sequential()
    # model.add(Flatten(input_shape=(size, size)))
    #
    # model.add(Dense(512,activation='relu'))
    # model.add(Dense(256,activation='relu'))
    # model.add(Dense(len(alph), activation="softmax"))
    # model.compile(optimizer=Adam(),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    model = keras.Sequential()
    model.add(
        Convolution2D(filters=64, kernel_size=(4, 4), padding='same', input_shape=(size, size, 1), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(4, 4), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(filters=128, kernel_size=(4, 4), padding='same', activation='relu'))
    model.add(Convolution2D(filters=128, kernel_size=(4, 4), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(alph), activation=tensorflow.nn.softmax))
    # model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    return model
    return model


def trainModel(model, fileName: str):
    X_train, x_train_cat, X_test, y_test_cat = prepareData()
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1,
                                                                factor=0.5,
                                                                min_lr=0.000001)
    model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction],
              batch_size=64, epochs=100)
    # model.fit(X_train, x_train_cat, epochs=10)


    # model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[CustomCallback()],
    #           batch_size=40, epochs=100)

    # train_generator, validation_generator = prepareData2()
    #
    # history = model.fit_generator(train_generator,
    #                        epochs=100,
    #                        verbose=1,
    #                        )

    model.save("models/" + fileName + ".h5")
    model.summary()

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()