import matplotlib.pyplot as plt
import numpy as np
import cv2

HEIGHT, WIDTH = 256, 256
img_list = ['camera', 'eyeR', 'head', 'salesman']
PLT_ROW, PLT_COL, plt_idx = 3, 2, 1
PLT_FONT_SIZE = 7
plt.rcParams.update({'font.size': PLT_FONT_SIZE})

def full_scale_contrast(src):
    max_v = np.max(src)
    min_v = np.min(src)
    if max_v == min_v:
        return src * 0
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


for img_name in img_list:
    curr_img = np.fromfile(file=f"{img_name}.bin", dtype=np.uint8).reshape(HEIGHT, WIDTH)
    plt_idx = 1
    showimg(src=curr_img, title=img_name, cmap='gray')
    plt_idx += 1

    fft2d_img = np.fft.fftshift(np.fft.fft2(curr_img))
    fft2d_img_fsc = full_scale_contrast(fft2d_img)
    showimg(src=np.real(fft2d_img_fsc), title=f"Re[DFT({img_name})]", cmap='gray')
    showimg(src=np.imag(fft2d_img_fsc), title=f"Im[DFT({img_name})]", cmap='gray')

    log_img = np.log(np.abs(fft2d_img) + 1)
    log_img_fsc = full_scale_contrast(src=log_img)
    showimg(src=log_img_fsc, title=f"Log-magnitude spectrum({img_name})", cmap='gray')

    phase_img = np.angle(fft2d_img)
    phase_img_fsc = full_scale_contrast(src=phase_img)
    showimg(src=phase_img_fsc, title=f"Phase of DPT({img_name})", cmap='gray')

    plt.show()
