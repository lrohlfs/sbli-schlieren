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


def coherence_plot(schlieren, x, y, lf_th=1000, fname='coherence_x140'):
    # plot previously calculated coherence in the low frequency domain
    i = int(np.argwhere(schlieren.f == lf_th))
    lf = np.sum(schlieren.coh[:i, :, :], axis=0)

    fig, ax1 = plt.subplots(1, 1, figsize=(9, 5))
    p1 = ax1.imshow(lf / i, cmap='Greys', vmin=-0.05, vmax=0.6)
    ax1.set_axis_off()
    ax1.plot(y, x, 'ro')
    fig.suptitle('Contour shows summed up Coherence')
    fig.colorbar(p1, ax=ax1, shrink=0.75)
    plt.tight_layout()
    plt.savefig(schlieren.path + fname, dpi=300)
    plt.close()


def coherence_line(schlieren, x, y_range):
    for y in y_range:
        schlieren.coherence(reference=schlieren.images[:, x, y], segments=1000, pixel=32)
        coherence_plot(schlieren, x, y, fname='coherence_y%03d_x%03d' % (x, y))
        print('Saved coherence_y%03d_x%03d.png' % (x, y))
    return
