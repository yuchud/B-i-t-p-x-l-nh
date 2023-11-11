import matplotlib.pyplot as plt
import numpy as np
import cv2

PLT_ROW, PLT_COL, plt_idx = 2, 2, 1
HEIGHT, WIDTH = 256, 256
TEMPLATE_HEIGHT, TEMPLATE_WIDTH = 47, 15
PLT_FONT_SIZE = 7


def showimg(img, is_histogram=False, title=None, axis=False, cmap=None):
    global PLT_ROW, PLT_COL, plt_idx
    plt.subplot(PLT_ROW, PLT_COL, plt_idx)
    plt_idx += 1
    if is_histogram:
        plt.plot(img)
    else:
        plt.imshow(img, cmap)
    plt.title(title)
    plt.axis(axis)


plt.rcParams.update({'font.size': PLT_FONT_SIZE})

org_img = np.fromfile(file="actontBin.bin", dtype=np.uint8).reshape(WIDTH, HEIGHT)
#org_img = cv2.cvtColor(src=org_img, code=cv2.COLOR_BGR2RGB)
showimg(img=org_img, title="origin img", cmap='gray')

template_t = np.zeros(shape=(TEMPLATE_HEIGHT, TEMPLATE_WIDTH)).astype(np.uint8)
template_t[10:16, :] = 255
template_t[16:37, 6:10] = 255
#template_t = cv2.cvtColor(src=template_t, code=cv2.COLOR_BGR2RGB)
showimg(img=template_t, title="Template T", cmap='gray')

img_j1 = np.zeros(shape=(HEIGHT, WIDTH))
#img_j1 = cv2.cvtColor(src=img_j1, code=cv2.COLOR_BGR2RGB)

for row in range(HEIGHT - TEMPLATE_HEIGHT):
    for col in range(WIDTH - TEMPLATE_WIDTH):
        compare_rectangle = org_img[row:row + TEMPLATE_HEIGHT, col:col + TEMPLATE_WIDTH]
        # img_j1[row, col] = np.sum(np.logical_and(compare_rectangle, template_t))
        img_j1[row, col] = np.sum(np.logical_not(np.logical_xor(compare_rectangle, template_t)))

showimg(img=img_j1, title="image J1", cmap='gray')

_, img_j2 = cv2.threshold(src=img_j1, thresh=670, maxval=1000, type=cv2.THRESH_BINARY)
showimg(img=img_j2, title="image J2", cmap='gray')

plt.show()
