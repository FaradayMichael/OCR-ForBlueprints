from model import *
from tools import *
import os
import numpy as np
import tst



def print_letter(result):
    letters = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    return letters[result]

def predicting(imgFile, model, size):
    image = keras.preprocessing.image

    letters = tst.getLettersFromImg(imgFile, size)




    #img = image.load_img(letters, target_size=(64, 64))



    #x = image.img_to_array(img)


    #print(letters[0].shape)

    # cv2.imshow("0", letters[0][1])
    # # cv2.imshow("1", letters[1])
    # # cv2.imshow("1", letters[2])
    # cv2.waitKey(0)


    result = ""
    predictRes = []
    for x in letters:


        #x = letters
        #
        # cv2.imshow("1", x[1])
        # cv2.waitKey(0)
        x = np.expand_dims(x[1], axis=0)
        x = np.reshape(x, (size, size, 1))



        images = np.vstack([x])
        #print(images.shape)
        images = x.reshape((1, 28, 28, 1))

        predictRes.append(model.predict(images))
        #resul = int(np.argmax(classes))

        # result += print_letter(resul)
        # result += " "
        #
        # print(print_letter(resul))

    return predictRes

if __name__ == '__main__':
    model = keras.models.load_model("models/MF_28_cln_v1.h5")

    #img = "images/s.png"

    # os.chdir("C:/wrk/cnn/Cyrillic64_cut/А")
    # test = []
    # for img in os.listdir():
    #     test.append(predicting(img, model))
    #
    # i=0
    # for p in test:
    #     print(p)
    #     if p == 'А':
    #         i=i+1
    #
    # print(i,len(test))
    # print(i/len(test))


    #lernNN(CreateModel_v1(), "MF_28_cln_v1")
    #lernNN(CreateModel_v3(), "M3_64_Adam_v3")
    #lernNN(keras.models.load_model('models/M1_64_cln_v1.h5'), "M1_64_cln_v1")



    p = predicting("images/s.png", model,28)
    print(p[0])
    print(print_letter(int(np.argmax(p[0]))))
    # for i in p:
    #     print(print_letter(int(np.argmax(i))))
    # print(predicting("C:/wrk/cnn/mai/LET.png", model=keras.models.load_model("models/M1_64_Adam_v2.h5"), size=64))
    # #print(predicting("C:/wrk/cnn/mai/LET.png", model=keras.models.load_model("models/M3_64_Adam_v3.h5"), size=64))
    #
    #print(open("images/t.txt", encoding='utf-8').readline())

    # for m in os.listdir("models"):
    #     model = keras.models.load_model("models/"+m)
    #     res = imgToStr(model, "images/s.png")
    #     print(m, res)

    # res = predicting("images/r.png", model,64)
    # print(res)

