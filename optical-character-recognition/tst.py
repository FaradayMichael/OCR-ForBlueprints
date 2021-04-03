import cv2
import numpy
import os
from shutil import copyfile


import cv2
import idx2numpy
import numpy
import os
from model import *
from tensorflow import keras








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


# def predicting(imgFile, model, size):
#     image = keras.preprocessing.image
#
#     letters = getLettersFromImg(imgFile, size)
#
#
#
#
#     #img = image.load_img(letters, target_size=(64, 64))
#
#
#
#     #x = image.img_to_array(img)
#
#
#     #print(letters[0].shape)
#
#     # cv2.imshow("0", letters[0][1])
#     # # cv2.imshow("1", letters[1])
#     # # cv2.imshow("1", letters[2])
#     # cv2.waitKey(0)
#
#
#     result = ""
#     predictRes = []
#     for x in letters:
#
#
#         #x = letters
#         #
#         # cv2.imshow("1", x[1])
#         # cv2.waitKey(0)
#         x = numpy.expand_dims(x[1], axis=0)
#         x = numpy.reshape(x, (size, size, 1))
#
#
#
#         images = numpy.vstack([x])
#         #print(images.shape)
#         images = x.reshape((1, 28, 28, 1))
#
#         predictRes.append(model.predict(images))
#         #resul = int(np.argmax(classes))
#
#         # result += print_letter(resul)
#         # result += " "
#         #
#         # print(print_letter(resul))
#
#     return predictRes

def emnist_predict_img(model, img):
    img_arr = numpy.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr[0] = numpy.rot90(img_arr[0], 3)
    img_arr[0] = numpy.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 64, 64, 1))

    result = model.predict_classes([img_arr])
    return chr(emnist_labels[result[0]])

def imgToStr(model, image_file: str):
    letters = getLettersFromImg(image_file)
    res = ""
    for i in range(len(letters)):
        dn = letters[i+1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        res += emnist_predict_img(model, letters[i][2])
        if (dn > letters[i][1]/4):
            res += ' '
    return res

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

# if __name__ == '__main__':
#     letters = getLettersFromImg("images/a1.png")
#
#     cv2.imshow("0", letters[0][2])
#     cv2.waitKey(0)



def ts(image_file:str, out_size=64):
    #Load img
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

    #img = cv2.imread(image_file, cv2.IMREAD_ANYCOLOR)
    trans_mask = img[:, :, 3] == 0
    img[trans_mask] = [255, 255, 255, 255]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, numpy.ones((3, 3), numpy.uint8), iterations=1)
    (hh, ww) = img_erode.shape[:2]
    img_erode = cv2.warpAffine(img_erode,cv2.getRotationMatrix2D((ww / 2, hh / 2),-10,1.0),(ww,hh))

    #get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    allRects = []
    maxArea = 0
    #get All Letters
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)

        if hierarchy[0][idx][3] == 0:
            #print(x, y, w, h)
            #cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            if x != 0 and y != 0:
                if w * h > maxArea:
                    maxArea = w * h
                allRects.append((x, y, w, h))
                cv2.imshow("a", img_erode[y:y+h, x:x+w])
                cv2.waitKey(0)


def tran():
    d1 = "C:\\Users\\farad\\Desktop\\OneDrive\\Dev\\cnn\\Cyrillic_cln_64_res"
    d2 = "C:\\Users\\farad\\Desktop\\OneDrive\\Dev\\cnn\\Cyrillic_cln_64"

    letters = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    dirs = os.listdir(d1)

    i = 0
    # for d in letters:
    #     j=0
    #     print(i)
    #     for img in os.listdir(d1+"/"+d):
    #         copyfile(d1+"/"+d+"/"+img, d2+"/"+str(i)+"/"+str(j)+".png")
    #         j+=1
    #         #os.system(str("cp "+d1+"/"+d+"/"+img+" "+d2+"/"+str(i)+"/"+img))
    #     i+=1
    for d in dirs:
        print(d)
        for img in os.listdir(d1+"/"+d):
            #print(d1+"\\"+d+"\\"+img)
            try:
                l = getLettersFromImg(d1+"\\"+d+"\\"+img)
                cv2.imwrite(str(d2+"/"+d+"/"+img),l[0][1])
            except IndexError:
                print(img)
                continue
        i+=1


def placeCenter(imgS:str):
    img = cv2.imread(imgS, cv2.IMREAD_UNCHANGED)
    trans_mask = img[:, :, 3] == 0
    img[trans_mask] = [255, 255, 255, 255]
    # cv2.imshow("a",img)
    # cv2.waitKey(0)
    h, w = img.shape[:2]
    nh = int(h*0.1)
    nw = int(w*0.1)
    blankImg = numpy.ones((h+nh*2,w+nw*2,4))

    blankImg[nh:h+nh,nw:w+nw] = img
    return blankImg


if __name__ == '__main__':
    # # #let = getLettersFromImg("images/r.png")
    #let = getLettersFromImg("C:\\Users\\Faraday\\Desktop\\OneDrive\\Dev\\cnn\\4.png")
    # o = 0
    # for l in let:
    # #     # cv2.imshow("a", l[1])
    # #     # cv2.waitKey(0)
    #     cv2.imwrite(str("C:\\Users\\Faraday\\Desktop\\OneDrive\\Dev\\cnn\\GOST\\"+str(o)+"\\"+str(o+33)+str(o+33)+".png"),l[1])
    #     o+=1
    # #ts("C:\\Users\\Faraday\\Desktop\\OneDrive\\Dev\\cnn\\2.png")
    # let = getLettersFromImg("C:\\Users\\Faraday\\Desktop\\OneDrive\\Dev\\cnn\\Cyrillic_cln_64_1\\0\\28.png")
    # cv2.imshow("a", let[2][1])
    # cv2.waitKey(0)
    tran()
    #placeCenter("images/a1.png")