import matplotlib.pyplot as plt
import numpy as np
import cv2

PLT_ROW, PLT_COL, plt_idx = 1, 2, 1
SIZE = 8
N, M = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
PLT_FONT_SIZE = 7
Uo, Vo = 1.5, 1.5


def full_scale_contrast(src):
    max_v = np.max(src)
    min_v = np.min(src)
    return (src - min_v) / (max_v - min_v) * 255


def showimg(src, is_histogram=False, title=None, cmap=None):
    global PLT_ROW, PLT_COL, plt_idx
    plt.subplot(PLT_ROW, PLT_COL, plt_idx)
    plt_idx += 1
    if is_histogram:
        plt.plot(src)
    else:
        plt.imshow(src, cmap)
        plt.axis(False)
    plt.title(title)


i5 = np.cos(2.0 * np.pi / 8.0 * (Uo * M + Vo * N))
showimg(src=full_scale_contrast(np.real(i5)), title="I5 Real", cmap='gray')
#showimg(src=full_scale_contrast(np.imag(i5)), title="i5 Imaginary", cmap='gray')

i5_fft2d = np.fft.fftshift(np.fft.fft2(i5))

i5_fft2d = np.round((i5_fft2d * 10.0 ** 4) * 10.0 ** (-4), 4)
print("Re[DFT(I5)]:")
print(np.real(i5_fft2d))

print("Im[DFT(I5)]:")
print(np.imag(i5_fft2d))
plt.show()
