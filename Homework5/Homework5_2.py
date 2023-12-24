import numpy as np
import matplotlib.pyplot as plt
import math

HEIGHT, WIDTH = 256, 256
PLT_ROW, PLT_COL, plt_idx = 1, 3, 1
PLT_FONT_SIZE = 7
plt.rcParams.update({'font.size': PLT_FONT_SIZE})


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


def MSE(a, b):
    return np.mean((a - b) ** 2)


# câu a
tiffany = np.fromfile(open('girl2.bin'), dtype=np.uint8).reshape(HEIGHT, WIDTH)
tiffany_noise_hi = np.fromfile(open('girl2Noise32Hi.bin'), dtype=np.uint8).reshape(HEIGHT, WIDTH)
tiffany_noise = np.fromfile(open('girl2Noise32.bin'), dtype=np.uint8).reshape(HEIGHT, WIDTH)

showimg(src=tiffany, title='Original Tiffany Image', cmap='gray')
showimg(src=tiffany_noise_hi, title='girl2Noise32Hi.bin Image', cmap='gray')
showimg(src=tiffany_noise, title='girl2Noise32.bin Image', cmap='gray')

MSE_tiffany_noise_hi = MSE(tiffany, tiffany_noise_hi)
MSE_tiffany_noise = MSE(tiffany, tiffany_noise)

print('Câu a:')
print(f'MSE girl2Noise32Hi.bin: {MSE_tiffany_noise_hi}')
print(f'MSE girl2Noise32.bin: {MSE_tiffany_noise}')
print('------')

plt.show()

# Câu b
U_cutoff = 64
U, V = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
HLtildeCenter = np.sqrt(U ** 2 + V ** 2) <= U_cutoff
HLtilde = np.fft.fftshift(HLtildeCenter)
PLT_ROW, PLT_COL, plt_idx = 1, 3, 1

filtered_tiffany = np.real(np.fft.ifft2(np.fft.fft2(tiffany) * HLtilde))
filtered_tiffany_noise_hi = np.real(np.fft.ifft2(np.fft.fft2(tiffany_noise_hi) * HLtilde))
filtered_tiffany_noise = np.real(np.fft.ifft2(np.fft.fft2(tiffany_noise) * HLtilde))

fsc_filtered_tiffany = full_scale_contrast(filtered_tiffany)
fsc_filtered_tiffany_noise_hi = full_scale_contrast(filtered_tiffany_noise_hi)
fsc_filtered_tiffany_noise = full_scale_contrast(filtered_tiffany_noise)

showimg(src=fsc_filtered_tiffany, title='Original Tiffany Image', cmap='gray')
showimg(src=fsc_filtered_tiffany_noise_hi, title='Filtered girl2Noise32Hi.bin Image', cmap='gray')
showimg(src=fsc_filtered_tiffany_noise, title='Filtered girl2Noise32.bin Image', cmap='gray')

MSE_filtered_tiffany = MSE(filtered_tiffany, tiffany)

MSE_filtered_tiffany_noise_hi = MSE(filtered_tiffany_noise_hi, tiffany)
ISNR_filtered_tiffany_noise_hi = 10.0 * np.log10(MSE_tiffany_noise_hi / MSE_filtered_tiffany_noise_hi)

MSE_filtered_tiffany_noise = MSE(filtered_tiffany_noise, tiffany)
ISNR_filtered_tiffany_noise = 10.0 * np.log10(MSE_tiffany_noise / MSE_filtered_tiffany_noise)

print('Câu b: ')
print(f'MSE Filtered: {MSE_filtered_tiffany}')
print(f'MSE Filtered girl2Noise32Hi.bin: {MSE_filtered_tiffany_noise_hi}')
print(f'ISNR Filtered girl2Noise32Hi.bin: {ISNR_filtered_tiffany_noise_hi}')
print(f'MSE Filtered girl2Noise32.bin: {MSE_filtered_tiffany_noise}')
print(f'ISNR Filtered girl2Noise32.bin: {ISNR_filtered_tiffany_noise}')
print('------')

plt.show()

# câu c
U_cutoff_H = 64
SigmaH = 0.19 * 256 / U_cutoff_H
U, V = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
HtildeCenter = np.exp((-2 * (math.pi ** 2) * (SigmaH ** 2)) / (256 ** 2) * (U ** 2 + V ** 2))
Htilde = np.fft.fftshift(HtildeCenter)
H = np.real(np.fft.ifft2(Htilde))
H2 = np.fft.fftshift(H)
ZPH2 = np.zeros((512, 512))
ZPH2[0:256, 0:256] = H2
PLT_ROW, PLT_COL, plt_idx = 1, 3, 1

DFT_ZPH2 = np.fft.fft2(ZPH2)

ZP_tiffany = np.zeros((512, 512))
ZP_tiffany[0:256, 0:256] = tiffany

ZP_tiffany_noise_hi = np.zeros((512, 512))
ZP_tiffany_noise_hi[0:256, 0:256] = tiffany_noise_hi

ZP_tiffany_noise = np.zeros((512, 512))
ZP_tiffany_noise[0:256, 0:256] = tiffany_noise

filtered_tiffany = np.real(np.fft.ifft2(np.fft.fft2(ZP_tiffany) * DFT_ZPH2))
filtered_tiffany = filtered_tiffany[128:384, 128:384]

filtered_tiffany_noise_hi = np.real(np.fft.ifft2(np.fft.fft2(ZP_tiffany_noise_hi) * DFT_ZPH2))
filtered_tiffany_noise_hi = filtered_tiffany_noise_hi[128:384, 128:384]

filtered_tiffany_noise = np.real(np.fft.ifft2(np.fft.fft2(ZP_tiffany_noise) * DFT_ZPH2))
filtered_tiffany_noise = filtered_tiffany_noise[128:384, 128:384]

fsc_filtered_tiffany = full_scale_contrast(filtered_tiffany)
fsc_filtered_tiffany_noise_hi = full_scale_contrast(filtered_tiffany_noise_hi)
fsc_filtered_tiffany_noise = full_scale_contrast(filtered_tiffany_noise)

showimg(src=fsc_filtered_tiffany, title='Original Tiffany Image', cmap='gray')
showimg(src=fsc_filtered_tiffany_noise_hi, title='Filtered girl2Noise32Hi.bin Image', cmap='gray')
showimg(src=fsc_filtered_tiffany_noise, title='Filtered girl2Noise32.bin Image', cmap='gray')

MSE_filtered_tiffany = MSE(filtered_tiffany, tiffany)

MSE_filtered_tiffany_noise_hi = MSE(filtered_tiffany_noise_hi, tiffany)
ISNR_filtered_tiffany_noise_hi = 10.0 * np.log10(MSE_tiffany_noise_hi / MSE_filtered_tiffany_noise_hi)

MSE_filtered_tiffany_noise = MSE(filtered_tiffany_noise, tiffany)
ISNR_filtered_tiffany_noise = 10.0 * np.log10(MSE_tiffany_noise / MSE_filtered_tiffany_noise)

print('Câu c: ')
print(f'MSE Filtered: {MSE_filtered_tiffany}')
print(f'MSE Filtered girl2Noise32Hi.bin: {MSE_filtered_tiffany_noise_hi}')
print(f'ISNR Filtered girl2Noise32Hi.bin: {ISNR_filtered_tiffany_noise_hi}')
print(f'MSE Filtered girl2Noise32.bin: {MSE_filtered_tiffany_noise}')
print(f'ISNR Filtered girl2Noise32.bin: {ISNR_filtered_tiffany_noise}')
print('------')

plt.show()

# Câu d
U_cutoff_H = 77.5
SigmaH = 0.19 * 256 / U_cutoff_H
U, V = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
HtildeCenter = np.exp((-2 * (math.pi ** 2) * (SigmaH ** 2)) / (256 ** 2) * (U ** 2 + V ** 2))
Htilde = np.fft.fftshift(HtildeCenter)
H = np.real(np.fft.ifft2(Htilde))
H2 = np.fft.fftshift(H)
ZPH2 = np.zeros((512, 512))
ZPH2[0:256, 0:256] = H2
PLT_ROW, PLT_COL, plt_idx = 1, 3, 1

DFT_ZPH2 = np.fft.fft2(ZPH2)

ZP_tiffany = np.zeros((512, 512))
ZP_tiffany[0:256, 0:256] = tiffany

ZP_tiffany_noise_hi = np.zeros((512, 512))
ZP_tiffany_noise_hi[0:256, 0:256] = tiffany_noise_hi

ZP_tiffany_noise = np.zeros((512, 512))
ZP_tiffany_noise[0:256, 0:256] = tiffany_noise

filtered_tiffany = np.real(np.fft.ifft2(np.fft.fft2(ZP_tiffany) * DFT_ZPH2))
filtered_tiffany = filtered_tiffany[128:384, 128:384]

filtered_tiffany_noise_hi = np.real(np.fft.ifft2(np.fft.fft2(ZP_tiffany_noise_hi) * DFT_ZPH2))
filtered_tiffany_noise_hi = filtered_tiffany_noise_hi[128:384, 128:384]

filtered_tiffany_noise = np.real(np.fft.ifft2(np.fft.fft2(ZP_tiffany_noise) * DFT_ZPH2))
filtered_tiffany_noise = filtered_tiffany_noise[128:384, 128:384]

fsc_filtered_tiffany = full_scale_contrast(filtered_tiffany)
fsc_filtered_tiffany_noise_hi = full_scale_contrast(filtered_tiffany_noise_hi)
fsc_filtered_tiffany_noise = full_scale_contrast(filtered_tiffany_noise)

showimg(src=fsc_filtered_tiffany, title='Original Tiffany Image', cmap='gray')
showimg(src=fsc_filtered_tiffany_noise_hi, title='Filtered girl2Noise32Hi.bin Image', cmap='gray')
showimg(src=fsc_filtered_tiffany_noise, title='Filtered girl2Noise32.bin Image', cmap='gray')

MSE_filtered_tiffany = MSE(filtered_tiffany, tiffany)

MSE_filtered_tiffany_noise_hi = MSE(filtered_tiffany_noise_hi, tiffany)
ISNR_filtered_tiffany_noise_hi = 10.0 * np.log10(MSE_tiffany_noise_hi / MSE_filtered_tiffany_noise_hi)

MSE_filtered_tiffany_noise = MSE(filtered_tiffany_noise, tiffany)
ISNR_filtered_tiffany_noise = 10.0 * np.log10(MSE_tiffany_noise / MSE_filtered_tiffany_noise)

print('Câu c: ')
print(f'MSE Filtered: {MSE_filtered_tiffany}')
print(f'MSE Filtered girl2Noise32Hi.bin: {MSE_filtered_tiffany_noise_hi}')
print(f'ISNR Filtered girl2Noise32Hi.bin: {ISNR_filtered_tiffany_noise_hi}')
print(f'MSE Filtered girl2Noise32.bin: {MSE_filtered_tiffany_noise}')
print(f'ISNR Filtered girl2Noise32.bin: {ISNR_filtered_tiffany_noise}')
print('------')

plt.show()
