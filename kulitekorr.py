# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:14:48 2021

@author: Lennart
"""
import sys
import threading
import time
from itertools import count

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, stats
from scipy.linalg import toeplitz
from numba import jit,prange



def foreach(f, l, threads=4, return_=False):
    """
    Apply f to each element of l, in parallel
    """

    if threads > 1:
        iteratorlock = threading.Lock()
        exceptions = []
        if return_:
            n = 0
            d = {}
            i = zip(count(), l.__iter__())
        else:
            i = l.__iter__()

        def runall():
            while True:
                iteratorlock.acquire()
                try:
                    try:
                        if exceptions:
                            return
                        v = next(i)
                    finally:
                        iteratorlock.release()
                except StopIteration:
                    return
                try:
                    if return_:
                        n, x = v
                        d[n] = f(x)
                    else:
                        f(v)
                except:
                    e = sys.exc_info()
                    iteratorlock.acquire()
                    try:
                        exceptions.append(e)
                    finally:
                        iteratorlock.release()

        threadlist = [threading.Thread(target=runall) for j in range(threads)]
        for t in threadlist:
            t.start()
        for t in threadlist:
            t.join()
        if exceptions:
            a, b, c = exceptions[0]
            raise (a, b, c)
        if return_:
            r = d.items()
            sorted(r)
            return [v for (n, v) in r]
    else:
        if return_:
            return [f(v) for v in l]
        else:
            for v in l:
                f(v)
            return


def parallel_map(f, l, threads=8):
    return foreach(f, l, threads=threads, return_=True)


f_s = 50000
path = 'I:/002Messdaten/Schlieren/Korrelation/'

data = np.squeeze(np.load(path + 'kulitekorr_1304.npy'))
# images = pims.ImageSequence(path+'img_*.tif')
# test = np.load(path+'schlieren_korr.npy')
small = np.load(path + 'small_array.npy')

test = small  # test[:,::2,::2]

orig_shape = small.shape

# test = np.reshape(test,(test.shape[0],test.shape[1]*test.shape[2])).swapaxes(1,0)
small = np.reshape(small, (small.shape[0], small.shape[1] * small.shape[2])).swapaxes(1, 0)

# test = []

# for image in images:
#     test.append(image)


# rows = range(0,test[0].shape[0])
# cols = range(0,test[0].shape[1])
segments = 2500

PSD = np.zeros((test[0].shape[0], test[0].shape[1], int(segments / 2 + 1)))
f_max = np.zeros((test[0].shape[0], test[0].shape[1]))


def mass_PSD(test):
    for i in rows:
        for j in cols:
            # filtered = signal.sosfilt(filt,test[:,i,j])
            f, Pxx = signal.welch(test[:, i, j], f_s, nperseg=segments, scaling='density')
            PSD[i, j] = (Pxx)
            f_max[i, j] = (f[(f * Pxx).argmax()])
            print(j + test[0].shape[1] * i)
    return PSD, f_max, f


def PSD_par(time_series):
    # filtered = signal.sosfilt(filt,test[:,i,j])

    f, Pxx = signal.welch(time_series, f_s, nperseg=segments, scaling='density')
    f_max = (f[(f * Pxx).argmax()])
    return Pxx, f_max, f


def Cohere_par(time_series):
    # filtered = signal.sosfilt(filt,test[:,i,j])

    f, Pxx = signal.coherence(time_series, data, fs=f_s, nperseg=segments)
    return Pxx


test_norm = test[1, :] / test[1, :].max()
test_norm2 = test[91536, :] / test[91536, :].max()
data_norm = data / data.max()

plt.plot(test_norm)
plt.plot(data_norm)
plt.plot(a)

# plt.cohere(test_norm, test_norm, Fs = 50000,NFFT = 4096)


r, _ = stats.pearsonr(test_norm2, data)

f, Cxx = signal.coherence(data[:-2], test[91536, 2:], fs=50000, nperseg=2048)
f1, Cxx1 = signal.coherence(data[:], test[91536, :], fs=50000, nperseg=2048)
f2, Cxx2 = signal.coherence(data[2:], test[91536, :-2], fs=50000, nperseg=2048)

a = signal.correlate(data, test[91536, :])

plt.semilogx(f, Cxx)
plt.semilogx(f1, Cxx1)
plt.semilogx(f2, Cxx2)

corr_cross = signal.correlate(test_norm, data_norm, mode='full')
corr_cross2 = signal.correlate(test_norm2, data_norm, mode='full')
corr_cross3 = signal.correlate(test_norm2, test_norm, mode='full')

corr1 = signal.correlate(data_norm, data_norm, mode='full')
corr2 = signal.correlate(test_norm, test_norm, mode='full')
corr3 = signal.correlate(test_norm2, test_norm2, mode='full')

lag = signal.correlation_lags(len(data_norm), len(test_norm))

corr_cross = corr_cross / np.max(corr_cross)
corr_cross2 = corr_cross2 / np.max(corr_cross2)
corr_cross3 = corr_cross3 / np.max(corr_cross3)

corr1 = corr1 / np.max(corr1)
corr2 = corr2 / np.max(corr2)
corr3 = corr3 / np.max(corr3)

plt.plot(corr1)
plt.plot(corr2)
plt.plot(corr3)

plt.plot(lag, corr_cross)
plt.plot(corr_cross2)
plt.plot(corr_cross3)

plt.semilogx(f, Cxx)

t0 = time.time()
output = parallel_map(PSD_par, test)
t1 = time.time()
print('Task done in%6.2f secs' % (t1 - t0))

t0 = time.time()
output = parallel_map(Cohere_par, test)
t1 = time.time()
print('Task done in%6.2f secs' % (t1 - t0))

c = [Cohere_par(j) for j in small]

t0 = time.time()
output = [PSD_par(ser) for ser in p_noise]
t1 = time.time()
print('Task done in%6.2f secs' % (t1 - t0))

PSD, f_max, f = zip(*output)
PSD = np.array(PSD)
f_max = np.array(f_max)
f = f[0]

lf_content = np.zeros(PSD.shape[0])
mf_content = np.zeros(PSD.shape[0])
hf_content = np.zeros(PSD.shape[0])

lf_threshhold = int(np.argwhere(f == 500))

mf_threshhold = int(np.argwhere(f == 10000))

for i in range(0, PSD.shape[0]):
    lf_content[i] = np.sum(PSD[i, 0:lf_threshhold])
    mf_content[i] = np.sum(PSD[i, lf_threshhold:mf_threshhold])
    hf_content[i] = np.sum(PSD[i, mf_threshhold:])

lf_img = lf_content.reshape((orig_shape[1], orig_shape[2]))
mf_img = mf_content.reshape((orig_shape[1], orig_shape[2]))
hf_img = hf_content.reshape((orig_shape[1], orig_shape[2]))

c = np.array(output)  # Coherence
lf_csd = np.zeros(c.shape[0])
mf_csd = np.zeros(c.shape[0])
hf_csd = np.zeros(c.shape[0])
for j in range(0, c.shape[0]):
    lf_csd[j] = np.sum(c[j, 0:lf_threshhold])
    mf_csd[j] = np.sum(c[j, lf_threshhold:mf_threshhold])
    hf_csd[j] = np.sum(c[j, mf_threshhold:])
lf_cimg = lf_csd.reshape((orig_shape[1], orig_shape[2]))
mf_cimg = mf_csd.reshape((orig_shape[1], orig_shape[2]))
hf_cimg = hf_csd.reshape((orig_shape[1], orig_shape[2]))

fig = plt.subplots(figsize=(12, 5))
plt.imshow(lf_co, cmap='hot', vmin=0.7, vmax=1)
plt.colorbar()
plt.title('Contour shows summed up Coherence up to 1000Hz per pixel, 100k images,50k fps, 10deg wedge')
plt.savefig(path + 'LF_Coherence_2.png', dpi=300)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

p1 = ax1.imshow(lf_img, cmap='nipy_spectral', vmin=-0.05, vmax=5)
ax1.set_title('LF Content (<1000Hz)')
ax2.imshow(mf_img, cmap='nipy_spectral', vmin=-0.05, vmax=5)
ax2.set_title('MF Content (1000Hz - 10000Hz)')
ax3.imshow(hf_img, cmap='nipy_spectral', vmin=-0.05, vmax=5)
ax3.set_title('HF Content (>10000Hz)')

cb_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
cbar = fig.colorbar(p1, cax=cb_ax, orientation='horizontal')

fig.suptitle('Contour shows summed up PSD content, based on 100k images captured at 50k fps, 10deg wedge')

plt.savefig(path + 'Energycontent_2.png', dpi=300)

f_max = f_max.reshape((orig_shape[1], orig_shape[2]))

fig = plt.subplots(figsize=(12, 5))
plt.imshow(f_max, cmap='hot', vmin=100, vmax=10000, interpolation='none')
plt.colorbar()
plt.title('Contour shows f*PSD Peak frequency per pixel, based on 100k images captured at 50k fps, 10deg wedge')
plt.savefig(path + 'f_max.png', dpi=300)

filtered1 = signal.sosfilt(filt, data)


# filtered2=signal.sosfilt(filt,test)

def Pearson_par(time_series):
    r, p = stats.pearsonr(time_series, data)
    return r, p


t0 = time.time()
output = parallel_map(Pearson_par, test)
t1 = time.time()
print('Task done in%6.2f secs' % (t1 - t0))

r = np.array(output)[:, 0].reshape((orig_shape[1], orig_shape[2]))
p = np.array(output)[:, 1].reshape((orig_shape[1], orig_shape[2]))

fig = plt.subplots(figsize=(12, 5))
plt.imshow(r, cmap='seismic', vmin=-0.15, vmax=0.15)
plt.colorbar()
plt.title('Contour shows Pearson Coefficient, 100k images,50k fps, 10deg wedge')
plt.savefig(path + 'pearson_corr.png', dpi=300)

ref_img = test[:, 1000].reshape((orig_shape[1], orig_shape[2]))

fig = plt.subplots(figsize=(12, 5))
plt.imshow(ref_img, cmap='Greys_r')
plt.colorbar()
plt.title('Reference Image, 100k images,50k fps, 10deg wedge')
plt.savefig(path + 'Reference.png', dpi=300)


def Wiener_opt(p_main, p_ref, L):
    Ns = p_main.size

    g_cc = signal.correlate(p_main, p_ref)
    g = g_cc[Ns - 1:Ns + L - 1]

    r_ac = signal.correlate(p_ref, p_ref)
    r = r_ac[Ns - 1:Ns + L - 1]

    RR = toeplitz(r.T)

    f = np.dot(np.linalg.inv(RR), g)

    p_noise = np.zeros(Ns)
    for n in range(L, Ns, 1):
        p_sum = 0
        for i in range(0, L, 1):
            p_sum = p_sum + f[i] * p_ref[n - (i + 1)]
        p_noise[n] = p_sum

    p_filtered = p_main - p_noise

    p_filtered[0:L - 1] = 0
    p_noise[0:L - 1] = 0
    print('computing done')
    return f, p_filtered[L:], p_noise[L:]


def Wiener_par(p_main):
    t0 = time.time()
    # p_main = p_main/p_main.max()
    Ns = p_main.size

    g_cc = signal.correlate(p_main, p_ref)
    g = g_cc[Ns - 1:Ns + L - 1]

    r_ac = signal.correlate(p_ref, p_ref)
    r = r_ac[Ns - 1:Ns + L - 1]

    RR = toeplitz(r.T)


    # def invert(RR,g):
    #     return np.dot(np.linalg.inv(RR),g)
    #
    # t0 = time.time()
    # f = invert(RR,g)
    # t1 = time.time()
    # print('Done in%6.2f secs' % (t1 - t0))


    f = np.linalg.solve(RR,g)

    p_noise = np.zeros(Ns)

    # p_sum = 0
    # for i in range(0,L,1):
    #     p_sum = p_sum+f[i]*p_ref[n-(i+1)]
    # @jit(nopython=True)
    # def loop(n):
    #     return np.sum(f * p_ref[n - L:n][::-1])

    # @jit(nopython=True)
    def calc(p_noise, f, p_ref, L, Ns):
        for n in range(L, Ns, 1):
            p_noise[n] = np.dot(f,p_ref[n - L:n][::-1])  # p_sum
        return p_noise

    t0 = time.time()
    p_noise = calc(p_noise, f, p_ref, L, Ns)
    t1 = time.time()
    print('Done in%6.2f secs' % (t1 - t0))



    # p_noise = parallel_map(loop,range(L,Ns,1))

    p_filtered = np.subtract(p_main, p_noise)

    # p_filtered[0:L-1] = 0
    # p_noise[0:L-1]    = 0
    t1 = time.time()
    print('Done in%6.2f secs' % (t1 - t0))
    return f, p_filtered[L:], p_noise[L:]


filt = signal.butter(50, 25000, 'lp', output='sos', fs=f_s)
filtered1 = signal.sosfilt(filt, data)

# p_main = test_norm2[100:]#test[91536,:]
p_ref = data[100:]
L = 5000

small = small[:, 100:]

# t0 = time.time()
# output = parallel_map(Wiener_par,p_main)
# t1 = time.time()
# print('Task done in%6.2f secs'% (t1-t0))


f,p_filtered,p_noise = Wiener_par(small[1746,:])

f,p_noise,p_filtered = zip(output)

t0 = time.time()

f_opt = np.zeros((len(small), L))
p_filtered = np.zeros((len(small), len(p_ref) - L))
p_noise = np.zeros((len(small), len(p_ref) - L))
i = 0
for main in small:
    f_opt[i, :], p_filtered[i, :], p_noise[i, :] = Wiener_par(main)
    i = i + 1
    print('Completed ' + str(i) + ' out of ' + str(len(small)))
t1 = time.time()
print('Task done in%6.2f secs' % (t1 - t0))

# p_noise1 = np.zeros(p_noise.shape)
# for i in range(len(p_noise)):
#     p_noise1[i,:] = p_noise[i,:]/p_noise[i,:].max()


p_noise = p_noise.reshape((128 * 224, 98900))

p_img2 = small.reshape((128, 224, 99900))
p_img3 = p_filtered.reshape((35, 100, 99400))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(p_img[:, :, 100])
ax2.imshow(p_img2[:, :, 600])
# t0 = time.time()
# p_sum = 0
# for i in range(0,L,1):
#     p_sum = p_sum+f[i]*p_ref[n-(i+1)]
# t1 = time.time()
# print('Task done in%6.2f secs'% (t1-t0))

export_arr = p_noise[:, 100:1000]
export_norm = np.zeros(export_arr.shape)
for i in range(len(p_noise)):
    export_norm[i, :] = export_arr[i, :] / export_arr[i, :].max()
export_norm = export_norm.reshape(35, 100, 900)

export_list = [export_norm[:, :, i] for i in range(0, 900)]

export_list2 = [p_noise[:,100:1000].reshape(35,100,900)[:, :, i] for i in range(0, 900)]

export_list = [noise_snippet[:, :, i] for i in range(0, 2000)]
export_list = [filt_snippet[:, :, i] for i in range(0, 2000)]
export_list = [img_snippet[:, :, i] for i in range(0, 2000)]
export = [img_f[i,:,:]+img_m for i in range(2000,3000)]
path = '/home/lennart/Data/'
import imageio

imageio.mimsave(path + 'filter_raw2000.gif', export, fps=100)


def multi_gif()
    #### GIF From Multiple Plots


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    for i in range(200):

        ax1.imshow(img_snippet[:,:,L+i],cmap = 'RdBu',vmin = -50, vmax = 50)
        ax2.imshow(noise_snippet[:,:,i],cmap = 'RdBu',vmin = -10, vmax = 10)
        ax3.imshow(filt_snippet[:,:,i],cmap = 'RdBu',vmin = -50, vmax = 50)
        plt.tight_layout()
        plt.savefig(path+'out/temp_'+format(i,'03d'))
        ax1.cla()
        ax2.cla()
        ax3.cla()
        print(i)

path = '/home/lennart/Data/out/'
fig,ax1 = plt.subplots(figsize = (6,5))
i = 0
for element in export:
    ax1.imshow(element,vmin = -20000,vmax = 20000,cmap = 'RdBu')
    plt.tight_layout()
    plt.savefig(path + 'temp_' + format(i, '03d'))
    ax1.cla()
    i=i+1
    print(i)


    import glob
    from PIL import Image
    fp_in = path+'temp_*.png'
    fp_out = path+'Wiener_Multi.gif'

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=100, loop=0)








plt.plot(p_img[20, 10, :])
plt.plot(p_img[20, 20, :])
plt.plot(p_img[20, 30, :])
plt.plot(p_img[20, 40, :])
plt.plot(p_img[20, 50, :])

f_opt, p_filtered, p_noise = Wiener_opt(p_main[3, :], p_ref, L)

plt.plot(p_main[3, :] / p_main[3, :].max())
plt.plot(p_noise / p_noise.max())
plt.plot(p_ref / p_ref.max())
plt.plot(p_filtered / p_filtered.max())

t0 = time.time()
output = parallel_map(Wiener_par, small)
t1 = time.time()
print('Task done in%6.2f secs' % (t1 - t0))

# w = [Wiener_par(j) for j in small]

w = np.array(output).swapaxes(1, 0)

w_img = w.reshape((orig_shape[0], int(orig_shape[1] / 4), int(orig_shape[2] / 4)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

p1 = ax1.imshow(small.reshape((int(orig_shape[1] / 4), int(orig_shape[2] / 4), orig_shape[0]))[:, :, 0], cmap='Greys_r')

ax2.imshow(w_img[0, :, :], cmap='Greys_r')

segments = 1000

# plt.subplots()
# plt.plot(filtered1*4000)
# plt.plot(filtered2)

f3, Pxx_spec3 = signal.welch(p_ref, f_s, nperseg=segments, scaling='density')

f1, Pxx_spec1 = signal.welch(p_main, f_s, nperseg=segments, scaling='density')
std1 = np.std(p_main)
f2, Pxx_spec2 = signal.welch(p_noise[4, :], f_s, nperseg=segments, scaling='density')
std2 = np.std(p_noise)
f4, Pxx_spec4 = signal.welch(p_noise1[4, :], f_s, nperseg=segments, scaling='density')

# plt.semilogx(f1*56/(1000*483),f1*Pxx_spec1/Pxx_spec1.max())
plt.semilogx(f2 * 56 / (1000 * 483), f2 * Pxx_spec2 / Pxx_spec2.max())
# plt.semilogx(f3*56/(1000*483),f3*Pxx_spec3/Pxx_spec3.max())
plt.semilogx(f4 * 56 / (1000 * 483), f4 * Pxx_spec4 / Pxx_spec4.max())

# cohere = signal.coherence(filtered1[0:len(filtered2)],filtered2,fs=f_s,nperseg=2500)

# plt.semilogx(cohere[0],cohere[1])

# c,f= plt.cohere(filtered1[0:len(filtered2)],filtered2,NFFT=1024,Fs=f_s)

# plt.semilogx(f,c)


plt.plot(p_noise/p_noise.max())
plt.plot(p_ref[L:]/p_ref.max())
plt.plot(p_main/p_main.max())

plt.plot(p_filt)

fc,cohere = signal.coherence(p_ref[L:],p_noise,fs = 50000,nperseg=2500)
plt.semilogx(fc,cohere)