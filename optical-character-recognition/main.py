from model import *
from tools import *
import os

if __name__ == '__main__':
    model = keras.models.load_model("models/M3_K5_Adam_v3.h5")

    #lernNN(CreateModel_v3(), "M3_K5_Adam_v3")
    #lernNN(keras.models.load_model('M3_K5_Adam_v1.h5'))

    for m in os.listdir("models"):
        model = keras.models.load_model("models/"+m)
        res = imgToStr(model, "images/s.png")
        print(m, res)

    # res = imgToStr(model,"images/s.png")
    # print(res)

