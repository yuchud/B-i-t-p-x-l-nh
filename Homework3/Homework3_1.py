import matplotlib.pyplot as plt
import numpy as np
import cv2

row, col, idx = 1, 3, 1
WIDTH, HEIGHT = 256, 256
THRESH = 95


def showimg(img, title=None, axis=False, cmap=None):
    global row, col, idx
    plt.subplot(row, col, idx)
    idx += 1
    plt.imshow(img, cmap)
    plt.title(title)
    plt.axis(axis)


def cau_a(img):
    _, thresh_img = cv2.threshold(src=img, thresh=THRESH, maxval=255, type=cv2.THRESH_BINARY)

    showimg(img=thresh_img, title="threshold img")


def cau_b(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh_img_gray = cv2.threshold(img_gray, THRESH, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(image=thresh_img_gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    contours_img = np.zeros(shape=(WIDTH, HEIGHT, 3), dtype=np.uint8)
    cv2.drawContours(image=contours_img, contours=contours, contourIdx=0, color=(255, 255, 255), thickness=1)

    showimg(img=contours_img, title="contours img", cmap='gray')


mammogram = np.fromfile(file="mammogram256.bin", dtype=np.uint8).reshape(WIDTH, HEIGHT)
mammogram = cv2.cvtColor(src=mammogram, code=cv2.COLOR_BGR2RGB)
showimg(img=mammogram, title="origin img")

cau_a(img=mammogram.copy())
cau_b(img=mammogram.copy())

plt.show()
