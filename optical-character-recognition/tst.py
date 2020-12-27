import cv2
import numpy
import os

def getLettersFromImg(imgFile, out_size=28):
    img = cv2.imread(imgFile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, numpy.ones((3, 3), numpy.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    rects = []

    for i in contours:
        [x, y, w, h] = cv2.boundingRect(i)
        rects.append((x, y, w, h))

    for j in range(len(rects)):
        #print(rects[j][0])
        if (rects[j][0]>=rects[j-1][0] and rects[j][0]<=rects[j-1][0]+rects[j-1][0]):
            pass


    #letters.sort(key=lambda x: x[0], reverse=False)
    #return letters

if __name__ == '__main__':
    getLettersFromImg("C:/wrk/a.png",64)
    # cv2.imshow("a", let[0][2])
    # cv2.imshow("a1", let[1][2])
    # cv2.imshow("a2", let[2][2])
    # cv2.waitKey(0)