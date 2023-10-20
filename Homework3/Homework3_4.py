import matplotlib.pyplot as plt
import numpy as np
import cv2

row, col, idx = 2, 2, 1
WIDTH, HEIGHT = 256, 256
PLT_FONT_SIZE = 7

def showimg(src, title=None, is_histogram=False, cmap=None):
    global row, col, idx
    plt.subplot(row, col, idx)
    idx += 1

    if is_histogram:
        plt.plot(src)
    else:
        plt.imshow(src, cmap)
        plt.axis(False)

    plt.title(title)


plt.rcParams.update({'font.size': PLT_FONT_SIZE})

johnny = np.fromfile(file="johnny.bin", dtype=np.uint8).reshape(WIDTH, HEIGHT)
johnny = cv2.cvtColor(src=johnny, code=cv2.COLOR_BGR2RGB)
johnny = cv2.cvtColor(src=johnny, code=cv2.COLOR_RGB2GRAY)

johnny_eq = cv2.equalizeHist(src=johnny)

hist_johnny = cv2.calcHist(images=[johnny], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
hist_johnny_eq = cv2.calcHist(images=[johnny_eq], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

showimg(src=johnny, title='Origin Img', cmap='gray')
showimg(src=johnny_eq, title='Histogram Equalized Image', cmap='gray')
showimg(src=hist_johnny, title='Histogram of Origin Image', is_histogram=True)
showimg(src=hist_johnny_eq, title='Histogram of Equalized Image', is_histogram=True)

plt.show()
