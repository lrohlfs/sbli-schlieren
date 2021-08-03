import numpy as np
import os
from matplotlib import pyplot as plt
from pco_tools import pco_reader as pco
from skimage.exposure import rescale_intensity


path = '/media/lennart/PIV_Auswertung/SG08_Day2_Glass/' #'/home/lennart/Downloads/piv-test/SG10_Glass/'
path_bg = '/home/lennart/Downloads/piv-test/SG10_bgsub/'
os.mkdir(path+'out/')
files = sorted(os.listdir(path))

# pco.load(path+'VN200.b16')


img = [pco.load(path+file) for file in files[:-1]]
img = np.array(img)
# bg = pco.load(path_bg+'VN010.b16')

img_mean = np.mean(img[:,:,:],axis = 0).astype('uint16')

# img_rescaled = rescale_intensity(img[10,:,:],in_range=(500,15000))

img_fluc = np.subtract(img,img_mean).astype('int16')

# plt.imshow(img[30,:,:],vmin = 500,vmax =15000)
# plt.imshow(img_mean)
# plt.imshow(img_fluc[31,:2160,:]-img_fluc[31,2160:,:])
# plt.imshow(img_fluc[31,:2160,:],vmin = -20000,vmax =20000)
#
#
# _,(ax1,ax2) = plt.subplots(1,2)
# ax1.imshow(img_fluc[i, :2160, :], vmin=-1500, vmax=1500)
# ax2.imshow(img[10,:2160,:]-img_mean[:2160,:],vmin = -1500,vmax =1500)
#



fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
for i in range(0, len(img), 1):
    ax1.imshow(img_fluc[i, :2160, :], vmin = -20000,vmax =20000,cmap='RdBu')
    plt.tight_layout()
    plt.savefig(path + 'out/temp_' + format(i, '04d'))
    ax1.cla()
    print(i)