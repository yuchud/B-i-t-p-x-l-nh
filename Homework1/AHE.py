import os.path

import cv2
import matplotlib.pyplot as plt
img_url = "dental.jpg"
if os.path.exists(img_url):
    img = cv2.imread(img_url, cv2.IMREAD_GRAYSCALE)

    plt.subplot(321)
    plt.imshow(img, cmap = 'gray')
    plt.title('Ảnh gốc')
    plt.axis(False)
    histg = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.subplot(322)
    plt.plot(histg)


    clahe8_8 = cv2.createCLAHE(tileGridSize=(8,8))
    img_ahe_8_8 = clahe8_8.apply(img)
    plt.subplot(323)
    plt.imshow(img_ahe_8_8, cmap = 'gray')
    plt.title('Ảnh với AHE có tiles 8x8')
    plt.axis(False)
    histg = cv2.calcHist([img_ahe_8_8], [0], None, [256], [0, 256])
    plt.subplot(324)
    plt.plot(histg)


    clahe16_16 = cv2.createCLAHE(tileGridSize=(16, 16))
    img_ahe_16_16 = clahe16_16.apply(img)
    plt.subplot(325)
    plt.imshow(img_ahe_16_16, cmap='gray')
    plt.title('Ảnh với AHE có tiles 16x16')
    plt.axis(False)
    histg = cv2.calcHist([img_ahe_16_16], [0], None, [256], [0, 256])
    plt.subplot(326)
    plt.plot(histg)

    plt.show()
else:
    print("Không tìm thấy ảnh")

