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
    img = "images/test.png"

    model1 = keras.models.load_model("models/M_28_A_v1.h5")
    model2 = keras.models.load_model("models/M_28_A_v2.h5")
    model3 = keras.models.load_model("models/M_28_A_v3.h5")


    # im = cv2.imread(img,cv2.IMREAD_UNCHANGED)
    # im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # print(im.shape)
    # res = predictLetter(im, model)
    # cv2.imshow("a", cv2.resize(im, (80, 80), interpolation=cv2.INTER_AREA))
    # cv2.waitKey(0)
    # print(res[-1][0:5])
    r1, r2, r3 = [],[],[]

    x = getLettersFromImg(img,28)
    for i in x:
#        r1.append(predictLetter(i[1], model1)[-1][0][0:2])
       # r2.append(predictLetter(i[1], model2)[-1][0][0:2])
        r3.append(predictLetter(i[1], model3)[-1][0:4])
        # res = predictLetter(i[1], model)
        # print(res[-1][0:3])



    # print(r1)
    # print(r2)
    for i in r3:
        print(i)
    #
    # print("//////////////////")
    # for i in x:
    #     res = predictLetter(i[1], model2)
    #     print(res[-1][0:5])

        #print(res[0])



