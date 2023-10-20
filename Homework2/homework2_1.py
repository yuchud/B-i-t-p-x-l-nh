import matplotlib.pyplot as plt
import numpy as np

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