import cv2
import numpy

#alph = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789."
alph = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"


def getLettersFromImg(image_file: str, out_size=28):
    ##Load img
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

    # img = cv2.imread(image_file)
    # trans_mask = img[:, :, 3] == 0
    # img[trans_mask] = [255, 255, 255, 255]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = placeCenter(gray)
    # cv2.imshow("a",gray)
    # cv2.waitKey()
    ret, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, numpy.ones((3, 3), numpy.uint8), iterations=1)
    # print(gray.shape)

    # get contours
    #img_erode = placeCenter(img_erode)
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    allRects = []
    maxArea = 0
    avgArea = 0
    u = 0

    # get All Letters
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)

        if hierarchy[0][idx][3] == 0:
            if x != 0 and y != 0:
                avgArea += w * h
                u += 1
                if w * h > maxArea:
                    maxArea = w * h
                allRects.append((x, y, w, h))
    # avgArea=avgArea/u
    subLettersRects = []
    rects = []
    letter_I_Ratio = 3

    ##Split on Letters/subLetters
    for r in allRects:
        if r[2] * r[3] < maxArea / 4:
            if r[3] / r[2] > letter_I_Ratio:
                rects.append(r)
            subLettersRects.append(r)
        else:
            rects.append(r)

    rects.sort(key=lambda x: x[0])
    letters = []
    k = -1
    for j in rects:
        k += 1
        # Check Ы
        if j[3] / j[2] < letter_I_Ratio:
            subs = []
            for i in subLettersRects:
                if i[0] >= j[0] - j[2] / 10 and i[0] + i[2] <= j[0] + j[2] + j[2] / 10:
                    # print(i[0], j[0])
                    subs.append(i)
            # Ё or Й
            if subs:
                allSubs = subs
                allSubs.append(j)
                minY = allSubs[0][1]
                maxX = 0
                for i in allSubs:
                    if i[0] + i[2] > maxX:
                        maxX = i[0] + i[2]
                    if i[1] < minY:
                        minY = i[1]
                l = gray[minY:j[1] + j[3], j[0]:j[0] + j[2]]
                letters.append((j[0], cv2.resize(l, (out_size, out_size), interpolation=cv2.INTER_AREA)))

            else:
                l = gray[j[1]:j[1] + j[3], j[0]:j[0] + j[2]]
                letters.append((j[0], cv2.resize(l, (out_size, out_size), interpolation=cv2.INTER_AREA)))
        else:
            letters.sort(key=lambda x: x[0])
            letters.remove(letters[-1])

            l = gray[rects[k - 1][1]:rects[k - 1][1] + rects[k - 1][3], rects[k - 1][0]:j[0] + j[2]]
            letters.append((rects[k - 1][0], cv2.resize(l, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    letters.sort(key=lambda x: x[0])
    return letters

def placeCenter(img):
    #img = cv2.imread(imgS, cv2.IMREAD_UNCHANGED)
    #trans_mask = img[:, :, 3] == 0
    #img[trans_mask] = [255, 255, 255, 255]
    # cv2.imshow("a",img[:,:])
    # cv2.waitKey(0)
    h, w = img.shape[0], img.shape[1]
    nh = int(h*0.1)
    nw = int(w*0.1)

    print(img.shape)
    blankImg =255* numpy.ones((h+nh*2,w+nw*2),numpy.uint8)

    blankImg[nh:h+nh,nw:w+nw] = img[:,:]
    return blankImg