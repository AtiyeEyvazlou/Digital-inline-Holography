'''
=========================================================================
  Computing simulation and reconstruction of digital in-line holograms
=========================================================================
This program contains one main stage of computing holography image and is
divided into the following steps:

Stage 1: loading data
Stage 2: computing wave propagation part (simulation)
Stage 3: computing wave back propagation part (reconstruction)
Stage 4: saving output as images
-------------------------------------------------------------------------
  Usage:
    calculation of wave propagation and back propagation of light using
    Kirchhoff-Fresnel formula utilized in inline holography.

-------------------------------------------------------------------------
  Inputs:
     dp                    :diffraction pattern
     support               :zero_padded parts or parts of the image you're sure
                            it has no usefull data like background
     illum_wavelen         :illumination wavelength in meters
     z                     :object to camera distance in meters
     cam_spacing           :pixel pitch in meters
     path                  :directory of the input pattern(image)
     n_iter                :iteration number
-------------------------------------------------------------------------
  Output:
    outputs are presented as inline plots
-------------------------------------------------------------------------
  Example call:
    display_output(dp , z) would get the input image as print the camera
    image, reconstructed image and error curves. However you could call
    following functions by yourself too.
    camera = generate(dp,z)
    rec = reconstruct(cam_abs, z)

-------------------------------------------------------------------------
  Reference(s):
    [1] Latychevskaia, T. and Fink, H.W., 2015. Practical algorithms for
        simulation and reconstruction of digital in-line holograms. Applied
        optics, 54(9), pp.2424-2434.
    [2] T. Latychevskaia and H.-W. Fink, "Solution to the twin image problem
        in holography," Physical review letters, vol. 98, no. 23, p. 233901,
        2007.
-------------------------------------------------------------------------
  Author:
     Atiyeh Eyvazlou (July 2023)
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft2, fftshift, ifft2
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d
import matplotlib.patches as patches

def forward_propagate(dp, z):  # eq 28

    x = np.arange(-dp.shape[0] / 2, dp.shape[0] / 2)
    y = np.arange(-dp.shape[1] / 2, dp.shape[1] / 2)
    X, Y = np.meshgrid(x, y)
    deltafx = 1 / (dp.shape[0] * cam_spacing)
    deltafy = 1 / (dp.shape[1] * cam_spacing)

    OB_spectrum = fftshift(fft2(fftshift(dp * deltafx)))
    aa = np.sqrt(1 - (illum_wavelen * X * deltafx) ** 2 - (illum_wavelen * Y * deltafy) ** 2)
    S = np.exp(1j * 2 * np.pi * z / illum_wavelen * aa)
    c = S * OB_spectrum
    d = fftshift(ifft2(fftshift(c)))
    return d


# %%

def back_propagate(cam_abs, z):  # eq 29

    a = fftshift(fft2(fftshift(cam_abs)))

    x = np.arange(-cam_abs.shape[0] / 2, cam_abs.shape[0] / 2)
    y = np.arange(-cam_abs.shape[1] / 2, cam_abs.shape[1] / 2)
    X, Y = np.meshgrid(x, y)
    deltafx = 1 / (cam_abs.shape[0] * cam_spacing)
    deltafy = 1 / (cam_abs.shape[1] * cam_spacing)
    aa = np.sqrt(1 - (illum_wavelen * X * deltafx) ** 2 - (illum_wavelen * Y * deltafy) ** 2)
    b = np.exp(-1j * 2 * np.pi * z / illum_wavelen * aa)
    c = a * b
    d = fftshift(ifft2(fftshift(c)))
    return d


# %%

def apply_cosine_window(image):
    rows, cols = image.shape[:2]
    window = np.hanning(rows)[:, np.newaxis] * np.hanning(cols)  # han : w(n) = 0.5 * (1 - cos(2πn / (N-1)))
    windowed_image = image * window
    return windowed_image


# %% psnr
def calculate_psnr(original_image, reconstructed_image):
    mse = np.mean((original_image - reconstructed_image) ** 2)
    # print(mse.shape)
    max_pixel_value = np.max(original_image)
    # print(max_pixel_value)
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr


# %%

def Latychevskaia_reconstruction(camera, support, n_iter):  # ref [2]

    errort = np.zeros(n_iter)
    errorphi = np.zeros(n_iter)

    cam_abs = np.abs(camera) ** 2
    cam_angle = np.angle(camera)

    x = np.arange(-cam_abs.shape[0] / 2, cam_abs.shape[0] / 2)
    y = np.arange(-cam_abs.shape[1] / 2, cam_abs.shape[1] / 2)
    X, Y = np.meshgrid(x, y)
    # set grid distances according to the pixel size
    deltafx = 1 / (cam_abs.shape[0] * cam_spacing)
    deltafy = 1 / (cam_abs.shape[1] * cam_spacing)
    X = X * deltafx
    Y = Y * deltafy
    r_s = np.sqrt(X ** 2 + Y ** 2 + z ** 2)

    phi = 2 * np.pi / illum_wavelen * r_s
    # phi = np.zeros(cam_abs.shape)
    b = np.abs(forward_propagate(np.ones(cam_abs.shape), z)) ** 2
    A0 = np.sqrt(cam_abs / b)
    # print(np.max(A0))
    plt.imshow(A0, cmap='gray')
    plt.title('A0')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    for i in range(n_iter):
        t = back_propagate(A0 * np.exp(1j * phi), z)
        abs_t = np.abs(t)
        angle_t = np.angle(t)

        abs_t[~support] = 1
        abs_t[abs_t > 1] = 1
        abs_t_new = abs_t
        plt.figure()
        plt.imshow(abs_t, cmap='gray')
        plt.title('abs_t' + ' in iteration number ' + str(i + 1))
        plt.axis('off')
        plt.colorbar()
        plt.show()
        # kernel = np.ones([5,5])
        # kernel[2,2]=4
        kernel = [[1, 4, 6, 4, 1],
                  [4, 16, 24, 16, 4],
                  [6, 24, 36, 24, 6],
                  [4, 16, 24, 16, 4],
                  [1, 4, 6, 4, 1]]
        for n in range(4):
            abs_t = convolve2d(abs_t, kernel, mode='same')

        t_new = abs_t * np.exp(1j * angle_t)
        u_new = forward_propagate(t_new, z)
        phi = 2 * np.pi % np.angle(u_new)

        errort[i] = np.sqrt(np.mean((abs_t_new - dp) ** 2))
        errorphi[i] = np.sqrt(np.mean((2 * np.pi % (phi - cam_angle)) ** 2))

        print('iteration = ' + str(i + 1))

    return abs_t_new, phi, errorphi, errort


# %%

def display_output(dp, cam_abs, abs_t, phi, errort, errorphi):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(4, 4), constrained_layout=True)
    # Add a rectangle patch around the image
    rect = patches.Rectangle((0, 0), dp.shape[0], dp.shape[1], linewidth=2, edgecolor='black', facecolor='none')
    rect1 = patches.Rectangle((0, 0), cam_abs.shape[0], cam_abs.shape[1], linewidth=2, edgecolor='black', facecolor='none')
    rect2 = patches.Rectangle((0, 0), phi.shape[0], phi.shape[1], linewidth=2, edgecolor='black', facecolor='none')
    rect3 = patches.Rectangle((0, 0), abs_t.shape[0], cam_abs.shape[1], linewidth=2, edgecolor='black', facecolor='none')
    # Add the rectangle patch to the axes
    
    ax = axes.ravel()
    ax[0].imshow(dp, cmap='gray')
    ax[0].set_title('Diffraction pattern')
    ax[0].axis('off')
    ax[0].add_patch(rect)
    ax[1].imshow(cam_abs, cmap='gray')
    ax[1].set_title('Camera')
    ax[1].axis('off')
    ax[1].add_patch(rect1)
    ax[2].imshow(phi, cmap='gray')
    ax[2].set_title('Reconstructed phase')
    ax[2].axis('off')
    ax[2].add_patch(rect2)
    ax[3].imshow(abs_t, cmap='gray')
    ax[3].axis('off')
    ax[3].set_title('Reconstructed Image')
    ax[3].add_patch(rect3)
    plt.show()

    print('rec image psnr:\t ' + str(calculate_psnr(dp, abs_t)))
    print('rec image ssim:\t ' + str(ssim(abs_t, dp, data_range=dp.max() - dp.min())))

    fig, axes = plt.subplots(ncols=1, nrows=2, constrained_layout=True)
    ax = axes.ravel()
    ax[0].plot(errorphi)
    ax[0].set_title('phase shifting distribution mismatch')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('E_phi')
    ax[1].plot(errort)
    ax[1].set_title('recovered distribution mismatch')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('E_t')

    return


# %% Reading diffraction pattern

illum_wavelen = 433e-9  # illumination wavelength in meters
z = 0.01  # in meters
cam_spacing = 10e-6  # pixel pitch in meters
n_iter = 31  # iteration number

path = r'F:\uni\M\Project\Atiye Eyvazlou\M.Sc. Simulations\Python Files\input\DH256.bmp'  # path to your using pattern image
dp = cv2.imread(path)  # read the image
dp = cv2.cvtColor(dp, cv2.COLOR_BGR2GRAY)  # turn the RGB image to grayscale one
dp = 1 - dp / 255

support = np.full(dp.shape, False)
Rx = dp.shape[0] / 4
Ry = dp.shape[1] / 4
for m in range(dp.shape[0]):
    for n in range(dp.shape[1]):
        x = dp.shape[0] / 2 - m
        y = dp.shape[1] / 2 - n
        if abs(x) < Rx and abs(y) < Ry:
            support[m, n] = True

# path = 'G:\Project\coding\pycharm projects\my_files\holo2.png'
# dp = cv2.imread(path)                                # read the image
# dp = cv2.cvtColor(dp, cv2.COLOR_BGR2GRAY)            # turn the RGB image to grayscale one
# dp1 = np.zeros( [dp.shape[0]*2,dp.shape[1]*2] )
# dp1[ int( dp.shape[0]/2) :int( 1.5* dp.shape[0]),int( dp.shape[1]/2) : int( 1.5* dp.shape[1]) ] = dp
# dp = 1 - dp1/255
# support = np.where(dp < 17, False , True )
# support = np.full(dp.shape, False)
# Rx = dp.shape[0] /4
# Ry = dp.shape[1] /4
# for m in range (dp.shape[0]) :
#     for n in range (dp.shape[1]):
#         x = dp.shape[0] / 2 - m
#         y = dp.shape[1] / 2 - n
#         if abs(x) < Rx  and abs(y) < Ry :
#         # if abs(x)**2 + abs(y)**2 < dp.shape[0]*100 :
#             support[m, n] = True

# support = np.where(dp==np.mean(dp), 1,0)


camera = forward_propagate(dp, z)

cam_abs = np.abs(camera)

abs_t, phi, errorphi, errort = Latychevskaia_reconstruction(camera, support, n_iter)

display_output(dp, cam_abs, abs_t, phi, errort, errorphi)
