import matplotlib.pyplot as plt
import numpy as np
import cv2

row, col, idx = 2, 2, 1
WIDTH, HEIGHT = 256, 256
PLT_FONT_SIZE = 7

def showimg(img, is_histogram=False, title=None, axis=False, cmap=None):
    global row, col, idx
    plt.subplot(row, col, idx)
    idx += 1
    if is_histogram:
        plt.plot(img)
    else:
        plt.imshow(img, cmap)
    plt.title(title)
    plt.axis(axis)

plt.rcParams.update({'font.size': PLT_FONT_SIZE})

lady = np.fromfile(file="lady.bin", dtype=np.uint8).reshape(WIDTH, HEIGHT)
lady = cv2.cvtColor(src=lady, code=cv2.COLOR_BGR2RGB)

hist = cv2.calcHist(images=[lady], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
showimg(img=lady, title="origin img")
showimg(img=hist, is_histogram=True, title="origin histogram", axis=True)

modified_lady = cv2.normalize(src=lady, dst=None, alpha = 0.0, beta = 255.0, norm_type=cv2.NORM_MINMAX)
modified_hist = cv2.calcHist(images=[modified_lady], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

showimg(img=modified_lady, title="modifed img")
showimg(img=modified_hist, is_histogram=True, title="modified histogram", axis=True)

plt.show()
