import os.path

import cv2
import matplotlib.pyplot as plt
img_url = "moon.jpg"
if os.path.exists(img_url):
    img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)

    plt.subplot(321)
    plt.imshow(img, cmap = 'gray')
    plt.title('Ảnh gốc')
    plt.axis(False)
    histg = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.subplot(322)
    plt.plot(histg)


    clahe5 = cv2.createCLAHE(clipLimit=5)
    img_clhe_5 = clahe5.apply(img)
    plt.subplot(323)
    plt.imshow(img_clhe_5, cmap = 'gray')
    plt.title('Ảnh với CLHE có limit = 5')
    plt.axis(False)
    histg = cv2.calcHist([img_clhe_5], [0], None, [256], [0, 256])
    plt.subplot(324)
    plt.plot(histg)


    clahe10 = cv2.createCLAHE(tileGridSize=(16, 16))
    img_clhe_10 = clahe10.apply(img)
    plt.subplot(325)
    plt.imshow(img_clhe_10, cmap='gray')
    plt.title('Ảnh với CLHE có limit = 10')
    plt.axis(False)
    histg = cv2.calcHist([img_clhe_10], [0], None, [256], [0, 256])
    plt.subplot(326)
    plt.plot(histg)

    plt.show()
else:
    print("Không tìm thấy ảnh")

