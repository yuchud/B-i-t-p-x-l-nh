import cv2
import matplotlib.pyplot as plt
import numpy as np

def lam_bai_1():
    w, h = 256, 256
    lena = np.fromfile("lena.bin", dtype = np.uint8).reshape((w, h))
    peppers = np.fromfile("peppers.bin", dtype = np.uint8).reshape((w, h))
    img_j = np.zeros((w, h))

    for r in range(0, h):
        for c in range(0, w//2):
            img_j[r][c] = lena[r][c]
        for c in range(w//2, w):
            img_j[r][c] = peppers[r][c]

    img_k = np.zeros((w, h))
    for r in range(0, h):
        for c in range(0, w//2):
            img_k[r][c] = img_j[r][w - c - 1]
            img_k[r][w-c-1] = img_j[r][c]

    plt.subplot(221)
    plt.imshow(lena, cmap='gray')
    plt.title("image Lena")
    plt.axis(False)

    plt.subplot(222)
    plt.imshow(peppers, cmap='gray')
    plt.title("image Peppers")
    plt.axis(False)

    plt.subplot(223)
    plt.imshow(img_j, cmap='gray')
    plt.title("image J")
    plt.axis(False)

    plt.subplot(224)
    plt.imshow(img_k, cmap = 'gray')
    plt.title("image k")
    plt.axis(False)

    plt.show()


def lam_bai_2():
    #print(help(cv2.imread))
    #print(help(cv2.imwrite))

    lenagray = cv2.imread("lenagray.jpg", cv2.IMREAD_GRAYSCALE)
    lenagray_neg = 255 - lenagray
    plt.subplot(121)
    plt.imshow(lenagray, cmap = 'gray')
    plt.title("Lena gray")
    plt.axis(False)

    plt.subplot(122)
    plt.imshow(lenagray_neg, cmap = 'gray')
    plt.title("Lena gray negative")
    plt.axis(False)

    plt.show()

    cv2.imwrite("../lenagray-neg.jpg", lenagray_neg)


lena512color = cv2.imread("lena512color.jpg")
lena512color = cv2.cvtColor(lena512color, cv2.COLOR_BGR2RGB)

plt.subplot(121)
plt.imshow(lena512color)
plt.title("origin")
plt.axis(False)

lena512color_modified = lena512color
lena512color_modified[:,:,1] = lena512color[:,:,0]
lena512color_modified[:,:,2] = lena512color[:,:,1]
lena512color_modified[:,:,0] = lena512color[:,:,2]

plt.subplot(122)
plt.imshow(lena512color_modified)
plt.title("modified")
plt.axis(False)

plt.show()

#lam_bai_1()
#lam_bai_2()

