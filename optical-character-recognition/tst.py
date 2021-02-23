import cv2
import numpy
import os
from shutil import copyfile

def getLettersFromImg_old(imgFile, out_size=64):
    img = cv2.imread(imgFile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, numpy.ones((3, 3), numpy.uint8), iterations=1)


    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    cv2.imshow("a",img_erode)
    cv2.waitKey(0)

    letters = []
    allRects = []

    maxArea = 0

    for i in contours:
        [x, y, w, h] = cv2.boundingRect(i)
        if x!=0 and y!=0:
            if w*h>maxArea:
                maxArea=w*h
            allRects.append((x, y, w, h))

    subLettersRects = []
    rects = []
    for r in allRects:
        if r[2]*r[3]<maxArea/10:
            subLettersRects.append(r)
        else:
            rects.append(r)
    #print(subLettersRects)

    letters = []
    for j in rects:
        if j[0]!=0 and j[1]!=0:
            #print(j)
            #cv2.rectangle(output, (rects[j][0], rects[j][1]), (rects[j][0] + rects[j][2], rects[j][1] + rects[j][3]), (70, 0, 0), 1)
            subs = []
            for i in subLettersRects:
                if i[0]>=r[0] and i[0]+i[2]<=r[0]+r[2]:
                    subs.append(i)
            if subs:
                allSubs = subs
                allSubs.append(j)
                minY = allSubs[0][1]
                maxX = 0
                for i in allSubs:
                    if i[0]+i[2]>maxX:
                        maxX=i[0]+i[2]
                    if i[1]<minY:
                        minY=i[1]
                l = img_erode[minY:j[1]+j[3], j[0]:j[0]+j[2]]
                letters.append((j[0],cv2.resize(l, (out_size, out_size), interpolation=cv2.INTER_AREA)))

            else:
                l = img_erode[j[1]:j[1] + j[3], j[0]:j[0] + j[2]]
                letters.append((j[0],cv2.resize(l, (out_size, out_size), interpolation=cv2.INTER_AREA)))
            print(j)




    letters.sort(key=lambda x: x[0])
    #sorted(letters, key=lambda x:x[0])
    return letters


def getLettersFromImg(image_file:str, out_size=64):
    #Load img
    #img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(image_file)

    #trans_mask = img[:, :, 3] == 0
    #img[trans_mask] = [255, 255, 255, 255]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, numpy.ones((3, 3), numpy.uint8), iterations=1)
    #print(gray.shape)

    #get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    allRects = []
    maxArea = 0
    avgArea = 0
    u = 0
    #get All Letters
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)

        if hierarchy[0][idx][3] == 0:
            #print(x, y, w, h)
            #cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            if x != 0 and y != 0:
                avgArea+=w*h
                u+=1
                if w * h > maxArea:
                    maxArea = w * h
                allRects.append((x, y, w, h))
    #avgArea=avgArea/u
    subLettersRects = []
    rects = []
    letbI = []
    #print(avgArea)
    #Split on Letters/subLetters
    bICH = 3
    for r in allRects:

        if r[2]*r[3]<maxArea/4:

            if r[3]/r[2]>bICH:
                # cv2.imshow("a", img_erode[r[1]:r[1] + r[3], r[0]:r[0] + r[2]])
                # cv2.waitKey(0)
                rects.append(r)
            # print(r[2],r[3])

            #print(r[0],r[1])
            subLettersRects.append(r)
        else:
            # print(r[0], r[1])
            # cv2.imshow("a", img_erode[r[1]:r[1] + r[3], r[0]:r[0] + r[2]])
            # cv2.waitKey(0)
            rects.append(r)

    rects.sort(key=lambda x:x[0])
    letters = []
    k = -1
    for j in rects:
        k+=1
        #Check Ы
        if j[3]/j[2]<bICH:
            #print(j)
            #cv2.rectangle(output, (rects[j][0], rects[j][1]), (rects[j][0] + rects[j][2], rects[j][1] + rects[j][3]), (70, 0, 0), 1)

            # cv2.imshow("a", img_erode[j[1]:j[1] + j[3], j[0]:j[0] + j[2]])
            # cv2.waitKey(0)
            subs = []
            for i in subLettersRects:
                if i[0]>=j[0]-j[2]/10 and i[0]+i[2]<=j[0]+j[2]+j[2]/10 :
                    #print(i[0], j[0])
                    subs.append(i)
            # Ё or Й
            if subs:
                allSubs = subs
                allSubs.append(j)
                minY = allSubs[0][1]
                maxX = 0
                for i in allSubs:
                    if i[0]+i[2]>maxX:
                        maxX=i[0]+i[2]
                    if i[1]<minY:
                        minY=i[1]
                l = img[minY:j[1]+j[3], j[0]:j[0]+j[2]]
                letters.append((j[0],cv2.resize(l, (out_size, out_size), interpolation=cv2.INTER_AREA)))

            else:
                l = img[j[1]:j[1] + j[3], j[0]:j[0] + j[2]]
                letters.append((j[0],cv2.resize(l, (out_size, out_size), interpolation=cv2.INTER_AREA)))
        else:
            letters.sort(key=lambda x: x[0])
            letters.remove(letters[-1])

            l=img[rects[k-1][1]:rects[k-1][1]+rects[k-1][3],rects[k-1][0]:j[0]+j[2]]
            letters.append((rects[k-1][0], cv2.resize(l, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    letters.sort(key=lambda x: x[0])
    return letters


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