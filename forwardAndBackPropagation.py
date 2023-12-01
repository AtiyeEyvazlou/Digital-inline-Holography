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
     illum_wavelen         :illumination wavelength in meters
     z = 0.01              :object to camera distance in meters
     cam_spacing           :pixel pitch in meters
     path                  :directory of the input pattern(image)
-------------------------------------------------------------------------
  Output:
    outputs are saved as Camera.png and ReconstructedImage.png
-------------------------------------------------------------------------
  Example call:
    camera = generate(dp,z) 
    rec = reconstruct(cam_abs, z)
-------------------------------------------------------------------------
  Reference(s):
    [1] Latychevskaia, T. and Fink, H.W., 2015. Practical algorithms for 
        simulation and reconstruction of digital in-line holograms. Applied 
        optics, 54(9), pp.2424-2434.
-------------------------------------------------------------------------
  Author:
     Atiyeh Eyvazlou (July 2023)
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fft2, fftshift, ifft2
import cv2
from skimage import io


def forward_propagate(dp, z):
    
    x = np.arange(-dp.shape[0]/2,dp.shape[0]/2)
    y = np.arange(-dp.shape[1]/2,dp.shape[1]/2)
    X,Y = np.meshgrid(x,y)
    deltafx = 1/(dp.shape[0]*cam_spacing)
    deltafy = 1/(dp.shape[1]*cam_spacing)
    
    OB_spectrum = fftshift(fft2(fftshift(dp * deltafx)))
    S = np.exp(-1j * np.pi * illum_wavelen * z * ((X * deltafx )** 2 + (Y * deltafy) ** 2))
    c = S * OB_spectrum
    d = fftshift(ifft2(fftshift(c)))
    return d

#%%

def back_propagate(dp, z):
    
    a = fftshift(fft2(fftshift(dp)))

    x = np.arange(-dp.shape[0]/2,dp.shape[0]/2)
    y = np.arange(-dp.shape[1]/2,dp.shape[1]/2)
    X,Y = np.meshgrid(x,y)
    deltafx = 1/(dp.shape[0]*cam_spacing)
    deltafy = 1/(dp.shape[1]*cam_spacing)
    
    b = np.exp(1j * np.pi * illum_wavelen * z * ((X * deltafx) ** 2 + (Y * deltafy) ** 2))
    c = a * b
    d = fftshift(ifft2(fftshift(c)))
    f = d.real + np.ones(d.shape)
    return f
#%%
# Reading diffraction pattern from local path
path = r'your/image/path/.../xyz.png'  # path to your using pattern image
dp = cv2.imread(path) # read the image
# dp = io.imread('https://github.com/AtiyeEyvazlou/phase-synchronization/blob/main/im1.png?raw=true')

dp = cv2.cvtColor(dp, cv2.COLOR_BGR2GRAY) # turn the RGB image to grayscale one



illum_wavelen = 633e-9 #illumination wavelength in meters
z = 0.01 # in meters
cam_spacing = 10e-6 #pixel pitch in meters


# generate the image pattern on the camera
camera = forward_propagate(dp,z) 
cam_abs = np.abs(camera)
# Reconstruct the image pattern from the camera image
rec = back_propagate(cam_abs, z)

# Display the result inline plot (reconstruction)
fig, axes = plt.subplots(ncols=3, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(dp, cmap='gray')
ax[0].set_title('Diffraction pattern')
ax[0].axis('off')
ax[1].imshow(cam_abs, cmap='gray')
ax[1].set_title('Camera')
ax[1].axis('off')
ax[2].imshow(rec, cmap='gray')
ax[2].set_title('reconstructed image')
ax[2].axis('off')
plt.show()

cv2.imwrite('im1.png', dp) 