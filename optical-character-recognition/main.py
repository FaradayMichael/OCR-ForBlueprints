from model import *
from tools import *
import os
import numpy as np



def print_letter(result):
    letters = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    return letters[result]

def predicting(imgFile, model, size):
    image = keras.preprocessing.image

    letters = getLettersFromImg(imgFile,size)



    #img = image.load_img(letters, target_size=(64, 64))



    #x = image.img_to_array(img)


    #print(letters.shape)

    # cv2.imshow("0", letters[0])
    # cv2.imshow("1", letters[1])
    # cv2.imshow("1", letters[2])
    # cv2.waitKey(0)


    result = ""

    for x in letters:
        #x = letters

        # cv2.imshow("1", x)
        # cv2.waitKey(0)
        x = np.expand_dims(x[1], axis=0)



        images = np.vstack([x])

        #images = x.reshape((1, 64, 64, 3))

        classes = model.predict(images, batch_size=1)
        resul = int(np.argmax(classes))
        result += print_letter(resul)
        result += " "

    return result

if __name__ == '__main__':
    #model = keras.models.load_model("models/M4_Adam_v2.h5")

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


    #lernNN(CreateModel_v1(), "M1_64_Adam_v2")
    #lernNN(CreateModel_v3(), "M3_64_Adam_v3")
    #lernNN(keras.models.load_model('models/M3_64_Adam_v2.h5'), "M3_64_Adam_v2")



    print(predicting("C:/wrk/cnn/mai/LET.png", model=keras.models.load_model("models/M3_64_Adam_v1.h5"), size=64))
    print(predicting("C:/wrk/cnn/mai/LET.png", model=keras.models.load_model("models/M1_64_Adam_v2.h5"), size=64))
    #print(predicting("C:/wrk/cnn/mai/LET.png", model=keras.models.load_model("models/M3_64_Adam_v3.h5"), size=64))

    print(open("images/t.txt", encoding='utf-8').readline())

    # for m in os.listdir("models"):
    #     model = keras.models.load_model("models/"+m)
    #     res = imgToStr(model, "images/s.png")
    #     print(m, res)

    # res = imgToStr(model,"images/r.png")
    # print(res)

