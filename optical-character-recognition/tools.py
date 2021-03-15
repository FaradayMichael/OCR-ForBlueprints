import cv2
import numpy


dir = "C:\\wrk\\cnn\\"
emnist_path = "C:\\wrk\\cnn\\emnist\\"

emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
                 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98,
                 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
                 118, 119, 120, 121, 122]


def getLettersFromImg(imgFile, out_size):
    img = cv2.imread(imgFile)
   # print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, numpy.ones((3, 3), numpy.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)



    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        #print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
        # hierarchy[i][0]: the index of the next contour of the same level
        # hierarchy[i][1]: the index of the previous contour of the same level
        # hierarchy[i][2]: the index of the first child
        # hierarchy[i][3]: the index of the parent
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = img[y:y + h, x:x + w]
            letters.append((x, cv2.resize(letter_crop, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    #         print(letter_crop.shape)
    #         cv2.imshow("0", letter_crop)
    #         cv2.waitKey(0)
    #
    #         # Resize letter canvas to square
    #         size_max = max(w, h)
    #         letter_square = 255 * numpy.ones(shape=[size_max, size_max], dtype=numpy.uint8)
    #
    #         if w > h:
    #             # Enlarge image top-bottom
    #             # ------
    #             # ======
    #             # ------
    #             y_pos = size_max // 2 - h // 2
    #             letter_square[y_pos:y_pos + h, 0:w] = letter_crop
    #         elif w < h:
    #             # Enlarge image left-right
    #             # --||--
    #             x_pos = size_max // 2 - w // 2
    #             letter_square[0:h, x_pos:x_pos + w] = letter_crop
    #         else:
    #             letter_square = letter_crop
    #
    #         # Resize letter to 28x28 and add letter and its X-coordinate
    #         #letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))
    #         letters.append(cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA))
    #
    #     # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=True)
    return reversed(letters)

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

# if __name__ == '__main__':
#     letters = getLettersFromImg("images/a1.png")
#
#     cv2.imshow("0", letters[0][2])
#     cv2.waitKey(0)