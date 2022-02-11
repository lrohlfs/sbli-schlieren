# Only run Calls to other functions from here!
from schlieren import Schlieren
from kulite import Kulite
from funtions import coherence_line, coherence_line_vert, phase_plot

from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
# Path with either npy file or subfolder with images
# path = 'D:/Arbeit_Homeoffice/Forschung/Schlieren/'
path = '/media/lennart/PIV_Auswertung/Schlieren/12_2021_Aktuator/SG+Ma2/'
# path = '/home/lennart/Schlieren/'
# path = 'E:/TSK_2021/Exports/'

fname = 'Ma2_SG10_3control.cine'
# fname = 'baseflow_red.npy'

# Initialize Classes
schlieren = Schlieren(path)
# kulite = Kulite(path)

# Load Schlieren Images
# Names: SG06_20000_settinvariation_4.cine, 40000_Test.cine
# schlieren.load_h5()
#
# schlieren.images_pa = schlieren.loadfromh5('images_pa')

schlieren.load(fname, fs=48000, start=720, end=72720, h5save=False)
# schlieren.img_mean = np.load(path+'baseflow_mean.npy')[128:,:]
# kulite.load('kulite_series.npy', end=10000)

# schlieren.img_fluc = np.subtract(schlieren.images, schlieren.img_mean)

# # Calculate Coherence to reference point in image as specified by x and y
# x = 215
# y_range = np.linspace(130, 330, 11).astype('uint16')
# coherence_line(schlieren, x, y_range)
#
# y=350
# x_range = np.linspace(180,240,7).astype('uint16')
# coherence_line_vert(schlieren, x_range, y)
# # schlieren.coherence_plot(fname='coherence_bubble_2')
#
# # Calculate and Plot Energy Spectrum
# schlieren.psd(pixel=16,segments=1024)
# schlieren.psd_plot(lf_th=750, mf_th=3000)

# ref point
# x=225
# y=182
# schlieren.spectral_complete(reference=schlieren.images[:, x, y], segments=960, pixel=16)
# schlieren.psd_plot(lf_th=1400, mf_th=6000)
# phase_plot(schlieren,x,y,lf_th = 700, mf_th = 3000)
# schlieren.coherence_plot()

# schlieren.compare_psd(x,y,x,y+10,segments=960)

# mask = np.ma.masked_less(schlieren.coh,0.25)

# for i in range(schlieren.images.shape[1]):
#     for j in range(schlieren.images.shape[2]):
#         for k in range(schlieren.coh.shape[0]):
#             if schlieren.coh[k,i,j] < 0.3: schlieren.phase[k,i,j] = 0

#
# edges = schlieren.canny_edge(par=(3, 30, 60))
# shockpos = []
# for edge in edges:
#     shockpos.append(np.argwhere(edge[180,:]==True)[0][0])
# f,pxx = signal.welch(shockpos,fs=5000,nperseg = 500)
# plt.semilogx(f,f*pxx)
#
#
# fig, ax1 = plt.subplots(figsize=(12, 5))
# i = 0
# for edge in edges:
#     ax1.imshow(edge[:250, :], cmap='Greys_r')
#     plt.tight_layout()
#     plt.savefig(path + 'edges/img_' + format(i, '03d') + '.png')
#     ax1.cla()
#     print(i)
#     i = i + 1

#
# def create_gif(path, sname='test.gif', duration=0.1):
#     import imageio
#     files = os.listdir(path)
#     imgs = []
#     for file in files:
#         imgs.append(imageio.imread(path + file))
#     imageio.mimsave(path + sname, imgs, duration=duration)
#
#
# create_gif(path + 'edges/')
#
# plt.imshow(edges[5])
# plt.show()
