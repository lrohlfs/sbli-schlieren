import os
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import cusignal
# import skimage.io as skio

from scipy import signal
from signal_analysis import wiener, wiener_gpu
path = '/home/lennart/Data/'
debug = 1
files = sorted(os.listdir(path+'Cine1/'))

kulite = np.load(path+'Kulite_Korr_Series.npy')[0, :]
# img_f = [plt.imread(path+'Cine1/'+file) for file in files[0:100000]]
#
# img_f = np.array(img_f,dtype='int16')
# img_m = np.mean(img_f,axis=0).astype('int16')
# img_f = np.subtract(img_f,img_m)
img_f = np.load(path+'Cine1_fluc.npy')


#plt.imshow(img[1000,:,:])


# img = img[:, ::2, ::2]
#

#
# p_ref = kulite
# noise = np.zeros(img.shape, dtype='f')
# filtered = np.zeros(img.shape, dtype='f')
# for i in range(img.shape[1]):
#     for j in range(img.shape[2]):
#         p_main = img[:, i, j]
#         filtered[L:, i, j], noise[L:, i, j] = wiener(p_main, p_ref, L)
#         print('Finished %d out of %d pixels' % (i * (img.shape[2]) + j + 1, img.shape[1] * img.shape[2]))



# img_gpu = cusignal.get_shared_mem((img.shape[0],pixel,pixel),dtype = img.dtype)
filt = signal.butter(50, 2000, 'lp', output='sos', fs=100000)
#signal.sosfilt(filt, data)
pixel = 16
segments = 2000
# img_gpu = cp.asarray(img[:,0:100,0:100])

pxx = cp.zeros((int(segments/2+1),img_f.shape[1],img_f.shape[2]),dtype = 'f')
cxx = cp.zeros((int(segments/2+1),img_f.shape[1],img_f.shape[2]),dtype = 'f')
ku = cp.tile(kulite[:len(img_f),None,None],(1,pixel,pixel))
# img_f = np.zeros(img_f.shape,dtype='int16')

#test = np.zeros(img.shape)

for i in range(int(img_f.shape[1]/pixel)):
    for j in range(int(img_f.shape[2]/pixel)):
        #test[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel]=img[:,i*pixel:(i+1)*pixel,j*pixel:(j+1)*pixel]

        img_gpu = img_f[:,i*pixel:(i+1)*pixel,j*pixel:(j+1)*pixel]
        f,pxx[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel]= cusignal.welch(img_gpu,fs = 100000,axis = 0,nperseg = segments)
        f, cxx[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = cusignal.coherence(img_gpu,ku,fs = 100000,axis = 0,nperseg = segments)
        # filtered = cusignal.sosfilt(filt,img_gpu,axis = 0)
        # img_f[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = filtered.get().astype('int16')
#
#
# #plt.plot(img[:,116,90])
#
# #f,pxx = signal.welch(img[:,116,90],fs = 100000,nperseg = 4000)
# #f,pxx = signal.welch(kulite,fs = 100000,nperseg = 4000)
# #plt.semilogx(f,f*pxx)
#
# #plt.imshow(pxx[1000,:,:].get())
#
psd = pxx.get()
csd = cxx.get()
#
f = f.get()
i=int(np.argwhere(f==1000))
j=int(np.argwhere(f==10000))
k=int(np.argwhere(f==25000))


fmax = np.max()

lf = np.sum(psd[:i,:,:],axis = 0)
mf = np.sum(psd[i:j,:,:],axis = 0)
hf = np.sum(psd[j:,:,:],axis = 0)

lfc = np.sum(csd[:i,:,:],axis = 0)
#


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

p1 = ax1.imshow(lf*100/lf.max(),cmap='nipy_spectral',vmin = 0,vmax=100)
ax1.set_title('LF Content (<1000Hz)')
ax2.imshow(mf*100/mf.max(),cmap='nipy_spectral',vmin = 0,vmax=100)
ax2.set_title('MF Content (1000Hz - 10000Hz)')
ax3.imshow(hf*100/hf.max(),cmap='nipy_spectral',vmin = 0,vmax=100)
ax3.set_title('HF Content (>10000Hz)')

cb_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
cbar = fig.colorbar(p1, cax=cb_ax, orientation='horizontal')

fig.suptitle('Contour shows summed up PSD content, based on 100k images captured at 100k fps, 10deg wedge')
plt.savefig(path + 'Energycontent.png', dpi=300)


plt.imshow(lfc/i,cmap='hot',vmin=0,vmax=0.3)

f_max = np.zeros((img_f.shape[1],img_f.shape[2]))
for i in range(img_f.shape[1]):
    for j in range(img_f.shape[2]):
        f_max[i,j] = (f[(f * psd[:,i,j]).argmax()])


kulite = (kulite-np.mean(kulite)).astype('f')
kulite = signal.sosfilt(filt,kulite)
overhead = 0
img_wn = np.zeros(img_f.shape,dtype='int16')
img_wf = np.zeros(img_f.shape,dtype='int16')

p_r = cp.asarray(kulite[overhead:len(img_f)]/kulite[overhead:len(img_f)].max())
p_r = cp.asarray(img_f[overhead:,100,5]/img_f[overhead:,100,5].max()).astype('f')


L = 2000
p_ref_array = np.zeros((len(p_r.get())-L,L),dtype = 'f')
i = 0
for n in range(L,len(p_r.get())):
    p_ref_array[i,:]=p_r.get()[n - L:n][::-1]
    i+=1

p_ref_array_cp = cp.array(p_ref_array)
if debug == 1:
    i = 0
    j = 0
    p_ref = p_r
else:
    for i in range(img_f.shape[1]):
        for j in range(img_f.shape[2]):
            p_main = cp.asarray(img_f[overhead:, i, j])
            img_wn[L+overhead:, i, j],img_wf[L+overhead:, i, j] = wiener_gpu(p_main, p_r, L,p_ref_array_cp)
        print('Finished %d out of %d pixels' % (i * (img_f.shape[2]) + j + 1, img_f.shape[1] * img_f.shape[2]))

L5000 = wiener_gpu(p_main, p_r, 10000).astype('int16')
L2000 = wiener_gpu(p_main, p_r, 5000).astype('int16')
L1000 = wiener_gpu(p_main, p_r, 1000).astype('int16')
L500 = wiener_gpu(p_main, p_r, 500).astype('int16')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
ax1.imshow(img_wn[5000,:,:],cmap='Greys')
ax2.imshow(img_f[5000,:,:],cmap='Greys')
ax3.imshow(img_wf[5000,:,:],cmap='Greys_r')

### Evaluate ####
#



def compare_psd(x1,y1,x2,y2):#
    pxx, _, f = psd(kulite / kulite.max(), 100000, 2000)
    # pxx,_,f=psd(img_f[:,x1,y1]/(img_wf[:,x1,y1].max()),100000,2000)
    pxx1,_,f=psd(img_f[:,x2,y2]/(img_wf[:,x2,y2].max()),100000,2000)

    fig,(ax1,ax2) = plt.subplots(2,1,figsize = (5,12))
    ax1.semilogx(f,4*f*pxx)
    ax1.semilogx(f,f*pxx1)


    ax2.loglog(f,4*pxx)
    ax2.loglog(f,pxx1)

compare_psd(120,97,119,97)



#
# #img_mean = np.mean(img,axis = 0)
# plt.imshow(lfc/i,cmap = 'hot')
#
# import imageio
# imageio.mimsave(path+'test.gif',img[0:100,:,:])
#
#
# img_gpu = cp.asarray(img).reshape(10000,160*256)
#
# C = ((cp.matmul(img_gpu,img_gpu.T))/(len(img_gpu-1))).astype('f')
#
# lam,A_s = cp.linalg.eigh(C)
#
# A_s = A_s.get()
#
#
# k = 9999
# U = np.matmul(A_s[:,k].reshape(10000,1),phi[:,k].reshape(1,40960))
# plt.imshow(U[0,:].reshape(160,256))
#
# U_arr = U.reshape(10000,160,256)
# imageio.mimsave('test2.gif',U_arr[0:1000,:,:])
