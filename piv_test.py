import numpy as np
import os
from matplotlib import pyplot as plt
from pco_tools import pco_reader as pco
from skimage.exposure import rescale_intensity


path = '/media/lennart/PIV_Auswertung/SG10_Day2_Glass/' #'/home/lennart/Downloads/piv-test/SG10_Glass/'
# path_bg = '/home/lennart/Downloads/piv-test/SG10_bgsub/'
if not os.path.isdir(path+'raw'):
    os.mkdir(path+'raw/')

if not os.path.isdir(path+'out'):
    os.mkdir(path+'out/')
files = sorted(os.listdir(path))

# pco.load(path+'VN200.b16')


img = [pco.load(path+file)[:2160,:] for file in files if file.endswith('b16')]
img = np.array(img)
# bg = pco.load(path_bg+'VN010.b16')

img_mean = np.mean(img[:,:,:],axis = 0).astype('uint16')

# img_rescaled = rescale_intensity(img[10,:,:],in_range=(500,15000))

img_fluc = np.subtract(img,img_mean).astype('int16')

plt.imshow(img_mean,vmin = 1000,vmax = 65000)
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

# fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
# for i in range(0, 480, 1):
#     ax1.imshow(u1[:,i].reshape(500,1280)+img_mean[1000:2000:2,::2], vmin = 1000,vmax =65000,cmap='jet')
#     plt.tight_layout()
#     plt.savefig(path + 'pod/temp_' + format(i, '04d'))
#     ax1.cla()
#     print(i)


fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
for i in range(0, len(img), 1):
    ax1.imshow(img[i, :2160, :], vmin = 1000,vmax =65000,cmap='Greys_r')
    plt.tight_layout()
    plt.savefig(path + 'raw/temp_' + format(i, '04d'))
    ax1.cla()
    print(i)

img_std = np.std(img,axis = 0)

fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
p1 = ax1.imshow(img_std,cmap = 'nipy_spectral',vmin = 1500,vmax =15000)
fig.colorbar(p1, ax=ax1, shrink=0.5)
plt.tight_layout()
plt.savefig(path + 'out/stddev')

import glob
from PIL import Image
fp_in = path + 'out/temp_*.png'
fp_out = path + 'fluc.gif'

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=100, loop=0)