import matplotlib.pyplot as plt
import numpy as np
import cv2

PLT_ROW, PLT_COL, plt_idx = 1, 2, 1
SIZE = 8
N, M = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
PLT_FONT_SIZE = 7
Uo, Vo = 2, 2


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


i2 = 0.5 * np.exp(-1j * 2.0 * np.pi / 8.0 * (Uo * M + Vo * N))
showimg(src=full_scale_contrast(np.real(i2)), title="I2 Real", cmap='gray')
showimg(src=full_scale_contrast(np.imag(i2)), title="I2 Imaginary", cmap='gray')

i2_fft2d = np.fft.fftshift(np.fft.fft2(i2))

i2_fft2d = np.round((i2_fft2d * 10**4) * 10**(-4), 4)
print("Re[DFT(I2)]:")
print(np.real(i2_fft2d))

print("Im[DFT(I2)]:")
print(np.imag(i2_fft2d))
plt.show()
