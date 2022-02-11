import h5py
import numpy as np

from signal_analysis import psd, wiener
from matplotlib import pyplot as plt
from scipy.signal import find_peaks_cwt

path = '/media/lennart/PIV_Auswertung/Schlieren/12_2021_Aktuator/SG+Ma2/'
h5file = 'Results.h5'

h5 = h5py.File(path+h5file, 'r+')
h5grp = 'Dez_2021/Control/Schlieren/'


images_pa = h5[h5grp+'images_pa']




closed = images_pa[890:1190,:,:]
open = images_pa[100:590,:,:]

closed_m = np.mean(closed,axis=0)
open_m = np.mean(open,axis=0)

pxx,f,_ = psd(closed[:,200,200]-closed_m[200,200],48000,960)
pxx,f,_ = psd(schlieren.img_fluc[:,200,200],48000,960)
plt.semilogx(f,f*pxx)


pxx,f,_ = psd(open[:,230,250],48000,960)
pxx,f,_ = psd(open[:,200,40],48000,segments=960)
plt.semilogx(f,f*pxx)

# Line Extraction:
x = 225
y = np.arange(50,301)
offset = 200
window = np.arange(176+50,225+50)
open_ml = open_m[x,y]
closed_ml = closed_m[x,y]
fig, ax1 = plt.subplots(figsize=(10,5))
i = offset
peak = []
for t in range(len(images_pa[offset:,x,y])):
    peaks = find_peaks_cwt(-images_pa[t+offset,x,window],6)
    peak.append(int(window[0]+peaks))
    # ax1.plot(open_m[225, 50:300])
    # ax1.plot(closed_m[225, 50:300])
    # ax1.plot(images_pa[t+offset,x,y])
    # ax1.plot(window[0]+peaks-50, images_pa[t+offset, x, window[peaks]], 'ro')
    # ax1.set_ylim(0, 200)
    # plt.tight_layout()
    # plt.savefig(path + 'transient/img_' + format(i, '04d') + '.png')
    # ax1.cla()
    print(i)
    window = np.arange(window[peaks]-24,window[peaks]+25)
    i = i + 1

tau = 1/1500

fit = 0.666+(0.958-0.666)*np.exp(-x*tau)
plt.plot(x,peak)
plt.plot(x+0.00869,fit)

print(1/(2*np.pi*tau))



plt.plot(open_m[225,50:300])
plt.plot(closed_m[225,50:300])




    # for data in schlieren.images_pa[:,225,50:300]:




######### Kulite Measurements ############

from scipy.signal import coherence, csd
from kulite import Kulite
path = '/media/lennart/PIV_Auswertung/Schlieren/01_2022_Aktuator/'

k = Kulite(path,h5grp = '2022_02_01/msg_001/')

k.load_h5()
k1 = k.data[0,:]
k2 = k.data[1,:]
t = k.data[2,:]

k1 = k1.reshape(200,1000)
k1_pa = np.mean(k1,axis=0)

k2 = k2.reshape(200,1000)
k2_pa = np.mean(k2,axis=0)

t = t.reshape(200,1000)
t_pa = np.mean(t,axis=0)

plt.figure(2)
plt.plot(k1_pa)
plt.plot(k2_pa)
plt.plot(t_pa/20+0.2)

x = np.linspace(0,1/50,1000)
Ma = (5*((k2_pa*0.7)**-0.286-1))**0.5

plt.figure(figsize=(12,8))

plt.subplot(2,1,1)
plt.plot(x,k2_pa*0.7,lw=2)
plt.xlabel('t[s]')
plt.ylabel('p/p_0')

plt.subplot(2,1,2)
plt.plot(x,Ma,lw=2)
plt.xlabel('t[s]')
plt.ylabel('Ma')




tau = 3500
fit = 0.215+(0.13-0.215)*np.exp(-x*tau)

plt.figure(figsize=(12,8))
plt.plot(x,k2_pa*0.7,lw=2)
plt.plot(x+0.01381,fit)
plt.axis([0,0.02,0.1,0.25])


k2f = k2-np.mean(k2)
k1f = k1-np.mean(k1)

sig,noi = wiener(k2f,k1f,10000)

p1,f,_ = psd(sig,50000,2500)
p2,f,_ = psd(noi,50000,2500)
pr,f,_ = psd(k2f,50000,2500)

plt.loglog(f,p1)
plt.loglog(f,p2)
plt.loglog(f,pr)


sig_pa = np.mean(sig.reshape(190,1000),axis=0)
noi_pa = np.mean(noi.reshape(190,1000),axis=0)

plt.plot(sig_pa)
plt.plot(noi_pa)
plt.plot(k2_pa-np.mean(k2))
plt.plot(k1_pa-np.mean(k1))
