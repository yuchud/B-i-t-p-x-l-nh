import matplotlib.pyplot as plt
import numpy as np
import cv2

row, col, idx = 1, 2, 1
WIDTH, HEIGHT = 256, 256


def showimg(img, is_histogram = False, title=None, axis=False, cmap=None):
    global row, col, idx
    plt.subplot(row, col, idx)
    idx += 1
    if is_histogram:
        plt.plot(img)
    else:
        plt.imshow(img, cmap)
    plt.title(title)
    plt.axis(axis)


lady = np.fromfile(file="lady.bin", dtype=np.uint8).reshape(WIDTH, HEIGHT)
lady = cv2.cvtColor(src=lady, code=cv2.COLOR_BGR2RGB)

hist = cv2.calcHist(images=[lady], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
showimg(img=lady, title="origin img")
showimg(img=hist, is_histogram=True, title="origin histogram", axis=True)

plt.show()
