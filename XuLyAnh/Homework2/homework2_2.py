import cv2
import matplotlib.pyplot as plt

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

cv2.imwrite("lenagray-neg.jpg", lenagray_neg)