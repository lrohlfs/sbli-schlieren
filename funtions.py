import numpy as np
from matplotlib import pyplot as plt


def psd_plot(schlieren, lf_th=1000, mf_th=10000, fname='energycontent.png'):
    # plot previously calculated per pixel psd split up in lf, mf, and hf content

    i = int(np.argwhere(schlieren.f == lf_th))
    j = int(np.argwhere(schlieren.f == mf_th))

    lf = np.sum(schlieren.pxx[:i, :, :], axis=0)
    mf = np.sum(schlieren.pxx[i:j, :, :], axis=0)
    hf = np.sum(schlieren.pxx[j:, :, :], axis=0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    p1 = ax1.imshow(lf, cmap='nipy_spectral', vmin=0, vmax=1000)
    ax1.set_title('LF Content (<%dHz)' % lf_th)
    fig.colorbar(p1, ax=ax1, shrink=0.75)
    p2 = ax2.imshow(mf, cmap='nipy_spectral', vmin=0, vmax=1000)
    ax2.set_title('MF Content (%dHz - %dHz)' % (lf_th, mf_th))
    fig.colorbar(p2, ax=ax2, shrink=0.75)
    p3 = ax3.imshow(hf, cmap='nipy_spectral', vmin=0, vmax=1000)
    ax3.set_title('HF Content (>%dHz)' % mf_th)
    fig.colorbar(p3, ax=ax3, shrink=0.75)

    # cb_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    # fig.colorbar(p1, cax=cb_ax, orientation='horizontal')

    fig.suptitle('Contour shows summed up PSD content')
    plt.tight_layout()
    # plt.savefig(schlieren.path + fname, dpi=300)

    return


def coherence_plot(schlieren, x, y, lf_th=1000, mf_th=10000, fname='coherence_x140'):
    # plot previously calculated coherence in the low frequency domain
    i = int(np.argwhere(schlieren.f == lf_th))
    j = int(np.argwhere(schlieren.f == mf_th))

    lf = np.sum(schlieren.coh[:i, :, :], axis=0)
    mf = np.sum(schlieren.coh[i:j, :, :], axis=0)

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))
    p1 = ax1.imshow(lf / i, cmap='hot', vmin=0, vmax=1)
    ax1.set_axis_off()
    ax1.plot(y, x, 'ro')
    fig.suptitle('Contour shows average LF Coherence')
    fig.colorbar(p1, ax=ax1, shrink=0.75)
    plt.tight_layout()
    plt.savefig(schlieren.path + fname + '_lf', dpi=300)
    plt.close()

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))
    p1 = ax1.imshow(mf / (j - i), cmap='hot', vmin=0, vmax=1)
    ax1.set_axis_off()
    ax1.plot(y, x, 'ro')
    fig.suptitle('Contour shows average MF Coherence')
    fig.colorbar(p1, ax=ax1, shrink=0.75)
    plt.tight_layout()
    plt.savefig(schlieren.path + fname + '_mf', dpi=300)
    plt.close()


def phase_plot(schlieren, x, y, lf_th=1000, mf_th=10000, fname='phaselag'):
    # plot previously calculated coherence in the low frequency domain
    cutoff = 200
    c = int(np.argwhere(schlieren.f == cutoff))
    i = int(np.argwhere(schlieren.f == lf_th))
    j = int(np.argwhere(schlieren.f == mf_th))

    lf = np.sum(schlieren.phase[c:i, :, :], axis=0)
    mf = np.sum(schlieren.phase[i:j, :, :], axis=0)

    lfc = np.sum(schlieren.coh[c:i, :, :], axis=0)
    mfc = np.sum(schlieren.coh[i:j, :, :], axis=0)

    mask = np.ma.array(lfc/(i-c))
    lf_masked = np.ma.masked_where(mask < 0.5, lf)
    mask = np.ma.array(mfc/(j-i))
    mf_masked = np.ma.masked_where(mask < 0.5, mf)

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))
    p0 = ax1.imshow(lfc / (i-c), cmap='Greys_r', vmin=0, vmax=0.5)
    p1 = ax1.imshow(lf_masked/(np.pi*(i-c)), cmap='seismic',vmin=-1, vmax=1)
    ax1.set_axis_off()
    ax1.plot(y, x, 'ro')
    fig.suptitle('Contour shows average LF phaselag ($-\pi:\pi$) with LF Coherence in Background')
    fig.colorbar(p1, ax=ax1, shrink=0.75)
    fig.colorbar(p0, ax=ax1, shrink=0.75)
    plt.tight_layout()
    plt.savefig(schlieren.path + fname + '_lf', dpi=300)
    plt.close()

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))
    p0 = ax1.imshow(mfc / (j-i), cmap='Greys_r', vmin=0, vmax=0.5)
    p1 = ax1.imshow(mf_masked / (np.pi*(j - i)), cmap='seismic',vmin=-1, vmax=1)
    ax1.set_axis_off()
    ax1.plot(y, x, 'ro')
    fig.suptitle('Contour shows average MF phaselag ($-\pi:\pi$) with MF Coherence in Background')
    fig.colorbar(p0, ax=ax1, shrink=0.75)
    fig.colorbar(p1, ax=ax1, shrink=0.75)
    plt.tight_layout()
    plt.savefig(schlieren.path + fname + '_mf', dpi=300)
    plt.close()


def coherence_line(schlieren, x, y_range):
    for y in y_range:
        schlieren.coherence(reference=schlieren.images[:, x, y], segments=1000, pixel=16)
        coherence_plot(schlieren, x, y, lf_th=562.5, mf_th=3000, fname='coherence_y%03d_x%03d' % (x, y))
        print('Saved coherence_y%03d_x%03d.png' % (x, y))
    return


def coherence_line_vert(schlieren, x_range, y):
    for x in x_range:
        schlieren.coherence(reference=schlieren.images[:, x, y], segments=960, pixel=16)
        schlieren.coherence_test(reference=schlieren.images[:, x, y], segments=512, pixel=16)
        coherence_plot(schlieren, x, y, fname='coherence_y%03d_x%03d' % (x, y))
        print('Saved coherence_y%03d_x%03d.png' % (x, y))
    return


def plot_imgseries(dataset, savepath, cmap='Greys_r', size=(8, 5),
                   vmin=None, vmax=None,
                   xrange=[None, None], yrange=[None, None]):
    if vmin == None: vmin = dataset.min()
    if vmax == None: vmax = dataset.max()

    fig, ax1 = plt.subplots(figsize=size)
    i = 0
    for data in dataset:
        ax1.imshow(data[xrange[0]:xrange[1], yrange[0]:yrange[1]],
                   cmap=cmap, vmin=vmin, vmax=vmax)
        plt.tight_layout()
        plt.savefig(savepath + '/img_' + format(i, '04d') + '.png')
        ax1.cla()
        print(i)
        i = i + 1



def create_gif(path, sname='test.gif', duration=0.1):
    files = sorted(listdir(path))
    imgs = []
    for file in files:
        imgs.append(imread(path + file))
    mimsave(path + sname, imgs[-400:800:10], duration=duration)

    return print('gif saved to: ' + path + sname)