import cv2
import numpy as np
import matplotlib.pyplot as plt

HEIGHT, WIDTH = 256, 256
PLT_ROW, PLT_COL, plt_idx = 1, 2, 1
PLT_FONT_SIZE = 7
plt.rcParams.update({'font.size': PLT_FONT_SIZE})

# square 7x7 cell 1/49
KERNEL7X7 = np.ones((7, 7)) / 49

salesman_img = np.fromfile(open('salesman.bin'), dtype=np.uint8).reshape(HEIGHT, WIDTH)


def full_scale_contrast(src):
    max_v = np.max(src)
    min_v = np.min(src)
    return np.round((src - min_v) / (max_v - min_v) * 255.0)


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


def convolve(_image, mode='full'):
    img_height, img_width = _image.shape[0], _image.shape[1]  # image size M x N
    kernel_height, kernel_width = KERNEL7X7.shape[0], KERNEL7X7.shape[1]  # kernel size m x n
    if mode == 'full':
        result = np.zeros((img_height + kernel_height - 1, img_width + kernel_width - 1))
        pad_image = np.zeros((img_height + kernel_height // 2 * 4, img_width + kernel_width // 2 * 4))
        pad_image[kernel_height // 2 * 2: kernel_height // 2 * 2 + img_height,
        kernel_width // 2 * 2: kernel_width // 2 * 2 + img_width] = _image

    elif mode == 'same':
        result = np.zeros((img_height, img_width))
        pad_image = np.zeros((img_height + kernel_height // 2 * 2, img_width + kernel_width // 2 * 2))
        pad_image[kernel_height // 2: kernel_height // 2 + img_height,
        kernel_width // 2: kernel_width // 2 + img_width] = _image

    elif mode == 'valid':
        result = np.zeros((max(img_height - kernel_height + 1, 0), max(img_width - kernel_width + 1, 0)))
        pad_image = _image
    else:
        return None

    result_height, result_width = result.shape[0], result.shape[1]
    for current_height in range(result_height):
        for current_width in range(result_width):
            window = pad_image[current_height:current_height + kernel_height,
                     current_width:current_width + kernel_width]
            result[current_height][current_width] = np.sum(window * KERNEL7X7)

    return result


# câu a
convolve_salesman_img = convolve(salesman_img, 'same')
fsc_convolve_salesman_img = full_scale_contrast(convolve_salesman_img)

showimg(src=salesman_img, title='Salesman Image', cmap='gray')
showimg(src=fsc_convolve_salesman_img, title='FullScaleContract Convolve Image', cmap='gray')

plt.show()

# câu b
H_image = np.zeros((HEIGHT // 2, WIDTH // 2))
H_image[62:69, 62:69] = 1 / 49
PLT_ROW, PLT_COL, plt_idx = 4, 2, 1

showimg(src=salesman_img, title='The original input image', cmap='gray')

# 383x383 có thể gây lỗi
# padded_image = np.zeros((383, 383))
padded_image = np.zeros((384, 384))
padded_image[0:HEIGHT, 0:WIDTH] = salesman_img
showimg(src=padded_image, title='The zero padded original image', cmap='gray')

padded_impulse_image = np.zeros((384, 384))
padded_impulse_image[0:HEIGHT // 2, 0:WIDTH // 2] = H_image
showimg(src=padded_impulse_image, title='The zero padded impulse response image', cmap='gray')

DFT_padded_image = np.fft.fft2(padded_image)
Log_magnitude_Spectrum_image = np.log(np.abs(np.fft.fftshift(DFT_padded_image)) + 1)
showimg(src=Log_magnitude_Spectrum_image,
        title='The centered DFT log-magnitude spectrum of the zero padded input image', cmap='gray')

DFT_padded_impulse_image = np.fft.fft2(padded_impulse_image)
Log_magnitude_Spectrum_H_image = np.log(np.abs(np.fft.fftshift(DFT_padded_impulse_image)) + 1)
showimg(src=Log_magnitude_Spectrum_H_image,
        title='The centered DFT log-magnitude spectrum of the zero padded impulse response image', cmap='gray')

DFT_padded_output_image = DFT_padded_image * DFT_padded_impulse_image
Log_magnitude_Spectrum_output_image = np.log(np.abs(np.fft.fftshift(DFT_padded_output_image)) + 1)
showimg(src=Log_magnitude_Spectrum_output_image,
        title='The centered DFT log-magnitude spectrum of the zero padded output image',
        cmap='gray')

padded_output_image = np.real(np.fft.ifft2(DFT_padded_output_image))
showimg(src=padded_output_image, title='The zero padded output image', cmap='gray')

final_image = padded_output_image[65:321, 65:321]
fsc_final_image = full_scale_contrast(final_image)
showimg(src=fsc_final_image, title='The final 256 × 256 output image', cmap='gray')

plt.show()

print(f'(b): max difference from part (a): {np.max(np.abs(fsc_final_image - fsc_convolve_salesman_img))}')

# câu c
H_image = np.zeros((HEIGHT, WIDTH))
H_image[125:132, 125:132] = 1 / 49
PLT_ROW, PLT_COL, plt_idx = 2, 2, 1

showimg(src=salesman_img, title='Original image', cmap='gray')
response_image256x256 = np.fft.fftshift(H_image)
showimg(src=response_image256x256, title='The 256 × 256 zero-phase impulse response image', cmap='gray')

response_image512x512 = np.zeros((512, 512))
response_image512x512[0:128, 0:128] = response_image256x256[0:128, 0:128]
response_image512x512[0:128, 384:512] = response_image256x256[0:128, 128:256]
response_image512x512[384:512, 0:128] = response_image256x256[128:256, 0:128]
response_image512x512[384:512, 384:512] = response_image256x256[128:256, 128:256]
showimg(src=response_image512x512, title='The 512 × 512 zero padded zero-phase impulse response image', cmap='gray')

padded_image = np.zeros((512, 512))
padded_image[0:HEIGHT, 0:WIDTH] = salesman_img

padded_output_image = np.fft.ifft2(np.fft.fft2(padded_image) * np.fft.fft2(response_image512x512))
final_image = np.real(padded_output_image)
final_image = final_image[0:256, 0:256]
final_image = full_scale_contrast(final_image)
showimg(src=final_image, title='The final 256 × 256 output image', cmap='gray')

plt.show()

print(f'(c): max difference from part (a): {np.max(np.abs(final_image - fsc_convolve_salesman_img))}')
