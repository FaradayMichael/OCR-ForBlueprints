from model import createModel_v1, createModel_v2, trainModel
from letters import getLettersFromImg, alph
from tensorflow import keras
import numpy
import cv2

size = 28

def predictLetter(imgArray, model, size=28):
    imgArray = 1-numpy.expand_dims(imgArray, axis=0) / 255.0
    imgArray = imgArray.reshape((1, size, size, 1))
    result = model.predict(x=imgArray)

    resultArray = []
    for i in range(len(alph)):
        resultArray.append((alph[i], round(result[0][i],3)))
    resultArray.sort(key=lambda x: x[1], reverse=True)

    #letter, predictions
    return alph[int(numpy.argmax(result))], resultArray

if __name__ == '__main__':
    img = "images/s.png"

    # model = createModel_v1(size)
    # trainModel(model, "MF_28")

    model = keras.models.load_model("models/MF_28.h5")

    x = getLettersFromImg(img,28)
    for i in x:

        res = predictLetter(i[1], model)
        print(res[-1][0:5])
        #print(res[0])



