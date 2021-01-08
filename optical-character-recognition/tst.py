import cv2
import numpy
import os

def getLettersFromImg(imgFile, out_size=64):
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


def testF(image_file:str,out_size=64):
    img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    trans_mask = img[:, :, 3] == 0
    img[trans_mask] = [255, 255, 255, 255]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, numpy.ones((3, 3), numpy.uint8), iterations=1)

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

    subLettersRects = []
    rects = []
    letbI = []
    for r in allRects:
        if r[2]*r[3]<maxArea/10:
            subLettersRects.append(r)
        else:
            rects.append(r)

    rects.sort(key=lambda x:x[0])
    letters = []
    k = -1
    for j in rects:
        k+=1
        if j[3]/j[2]<5:
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
        else:
            letters.sort(key=lambda x: x[0])
            letters.remove(letters[-1])

            l=img_erode[rects[k-1][1]:rects[k-1][1]+rects[k-1][3],rects[k-1][0]:j[0]+j[2]]
            letters.append((rects[k-1][0], cv2.resize(l, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    letters.sort(key=lambda x: x[0])
    # sorted(letters, key=lambda x:x[0])
    return letters


if __name__ == '__main__':
    let = testF("C:/wrk/hhh.png")
    for l in let:
        cv2.imshow("a", l[1])
        cv2.waitKey(0)
   # testF()
