import numpy as np
from signal_analysis import psd, wiener
from matplotlib import pyplot as plt
import time


def load_numpy(fname):
    array = np.load(fname)
    return array, array.shape


def schlieren_psd():
    global shape
    path = 'I:/002Messdaten/Schlieren/Korrelation/'
    f_s = 50000
    segments = 2500

    images, shape = load_numpy(path + 'small_array.npy')
    images = images.reshape((shape[0], int(shape[1] * shape[2]))).swapaxes(1, 0)
    f_max = np.zeros(int(shape[1] * shape[2]))
    pxx = np.zeros((int(shape[1] * shape[2]), int(segments / 2) + 1))
    i = 0
    for image in images:
        pxx[i, :], f_max[i], f = psd(image, f_s, segments)
        i = i + 1
        print(i)
    return f_max, pxx, f


def schlieren_energy(PSD, f):
    lf_content = np.zeros(PSD.shape[0])
    mf_content = np.zeros(PSD.shape[0])
    hf_content = np.zeros(PSD.shape[0])

    lf_threshold = int(np.argwhere(f == 1000))
    mf_threshold = int(np.argwhere(f == 10000))

    for i in range(0, PSD.shape[0]):
        lf_content[i] = np.sum(PSD[i, 0:lf_threshold])
        mf_content[i] = np.sum(PSD[i, lf_threshold:mf_threshold])
        hf_content[i] = np.sum(PSD[i, mf_threshold:])

    lf_img = lf_content.reshape((shape[1], shape[2]))
    mf_img = mf_content.reshape((shape[1], shape[2]))
    hf_img = hf_content.reshape((shape[1], shape[2]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    p1 = ax1.imshow(lf_img, cmap='nipy_spectral', vmin=0, vmax=100)
    ax1.set_title('LF Content (<1000Hz)')
    ax2.imshow(mf_img, cmap='nipy_spectral', vmin=0, vmax=100)
    ax2.set_title('MF Content (1000Hz - 10000Hz)')
    ax3.imshow(hf_img, cmap='nipy_spectral', vmin=0, vmax=100)
    ax3.set_title('HF Content (>10000Hz)')

    cb_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    cbar = fig.colorbar(p1, cax=cb_ax, orientation='horizontal')
    fig.suptitle('Contour shows summed up PSD content, based on 100k images captured at 50k fps, 10deg wedge')
    return lf_img, lf_content


def schlieren_wiener():
    global shape
    path = 'I:/002Messdaten/Schlieren/Korrelation/'
    f_s = 50000
    segments = 2500

    images, shape = load_numpy(path + 'schlieren_fluc_reduced.npy')
    images, shape = load_numpy(path + 'small_array.npy')
    images = images.reshape((shape[0], int(shape[1] * shape[2]))).swapaxes(1, 0)
    # images = images[:,::2,::2].reshape((shape[0], int(shape[1]/2 * shape[2]/2))).swapaxes(1, 0)
    p_ref = np.squeeze(np.load(path + 'kulitekorr_1304.npy'))
    p_ref = p_ref[0:50000] - np.mean(p_ref)
    L = 1000
    t0 = time.time()

    # f_opt = np.zeros((len(images), L))
    # p_filtered = np.zeros((len(images), len(p_ref) - L))
    # p_noise = np.zeros((len(images), len(p_ref) - L))
    output = []
    i = 0
    for main in images:
        output.append(wiener(main, p_ref, L))
        i = i + 1
        print('Completed ' + str(i) + ' out of ' + str(len(images)))
    t1 = time.time()
    print('Task done in %6.2f secs' % (t1 - t0))

    # output = [wiener(p_main,p_ref,L) for p_main in images]
    # output = wiener(images[1746,:],p_ref,L)
    return output


def calc_mode(k):
    U1 = np.matmul(A[:, k].reshape(10000, 1), phi_n[:, k].reshape(1, 28672))
    fig = plt.subplots(figsize=(6, 5))
    U_m = np.mean(U1, axis=0)
    plt.imshow(U_m.reshape(128, 224), cmap='RdBu', vmin=-100, vmax=100)
    plt.tight_layout()
    plt.savefig(path+'PODk' + str(k))
    # for i in range(200):
    #     plt.imshow(U1[i, :].reshape(128, 224), cmap='RdBu',vmin = -1000,vmax = 1000)
    #     if i == 0:
    #         plt.colorbar()
    #     plt.tight_layout()
    #     plt.savefig(path + 'out/temp_' + format(i, '03d'))
    #     plt.cla()
    #     print(i)
    #
    # import glob
    # from PIL import Image
    # fp_in = path + 'out/temp_*.png'
    # fp_out = path + 'PODk' + str(k) + '.gif'
    #
    # # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    # img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    # img.save(fp=fp_out, format='GIF', append_images=imgs,
    #          save_all=True, duration=100, loop=0)


def multi_gif():
    #### GIF From Multiple Plots

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    # for i in range(200):
    #     ax1.imshow(img_snippet[:, :, L + i], cmap='RdBu', vmin=-50, vmax=50)
    #     ax2.imshow(noise_snippet[:, :, i], cmap='RdBu', vmin=-10, vmax=10)
    #     ax3.imshow(filt_snippet[:, :, i], cmap='RdBu', vmin=-50, vmax=50)
    #     plt.tight_layout()
    #     plt.savefig(path + 'out/temp_' + format(i, '03d'))
    #     ax1.cla()
    #     ax2.cla()
    #     ax3.cla()
    #     print(i)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
    for i in range(200):
        ax1.imshow(U1[i, :].reshape(128, 224), cmap='RdBu', vmin=-25, vmax=25)
        plt.tight_layout()
        plt.savefig(path + 'out/temp_' + format(i, '03d'))
        ax1.cla()
        print(i)

    import glob
    from PIL import Image
    fp_in = path + 'out/temp_*.png'
    fp_out = path + 'PODk2.gif'

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=100, loop=0)

calc_mode(1)
if __name__ == '__main__':
    # f_max, pxx, f = schlieren_psd()
    # img,lst = schlieren_energy(pxx,f)

    output = schlieren_wiener()
    # f,p_filt,p_noise = zip(*output)
    #
    # p_noise = np.array(p_noise)
    # plt.imshow(f_max.reshape((35, 100)))
    # plt.show()
    # plt.plot(images[1746, L:])
    # plt.plot(p_filt[1746,:])
    #
    # plt.plot(p_noise / p_noise.max())
    # plt.plot(p_ref[L:] / p_ref.max())
    # plt.plot(images_fluc[1940, L:]/images_fluc[1940, L:].max())
    # plt.plot(output[1940,0,:]/output[1940, 0,:].max())
    #
    # plt.plot(p_ref[L:]/p_ref[L:].max())
