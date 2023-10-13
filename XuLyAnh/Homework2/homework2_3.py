import cv2
import matplotlib.pyplot as plt

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