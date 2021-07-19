import numpy as np
import os
from matplotlib import pyplot as plt
from numpy import ndarray


class Schlieren(object):
    images: ndarray

    def __init__(self, path):

        # Definitions
        self.path = path
        self.fs = 100000
        self.gpu = 1
        self.pxx = np.array([])
        self.f = np.array([])
        self.coh = np.array([])

        return

    def load(self, name, start=0, end=None, fs=100000):
        # Load schlieren images either from npy file or loose files in folder.
        try:
            self.images = np.load(self.path + name)

        finally:
            files = sorted(os.listdir(self.path + name))[start:end]
            self.images = np.array([plt.imread(self.path + name + file) for file in files], dtype='int16')
            img_mean = np.mean(self.images, axis=0, dtype='int16')
            self.images = np.subtract(self.images, img_mean)
        self.fs = fs
        return

    def psd(self, segments=2000, pixel=16):
        # calculate per pixel psd for entire image. Uses gpu if available
        try:
            import cupy as cp
            import cusignal

        finally:
            gpu = 0

        if gpu == 0:
            pxx_gpu = cp.zeros((int(segments / 2 + 1), self.images.shape[1], self.images.shape[2]), dtype='f')
            f_gpu = cp.zeros(int(segments / 2 + 1), dtype='f')

            for i in range(int(self.images.shape[1] / pixel)):
                for j in range(int(self.images.shape[2] / pixel)):
                    img_gpu = cp.asarray(self.images[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])
                    f_gpu, pxx_gpu[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = cusignal.welch(
                        img_gpu,
                        fs=self.fs,
                        axis=0,
                        nperseg=segments)
            self.pxx = pxx_gpu.get()
            self.f = f_gpu.get()

            return

        else:
            from scipy import signal
            pxx = np.zeros((int(segments / 2 + 1), self.images.shape[1], self.images.shape[2]), dtype='f')
            f = np.zeros(int(segments / 2 + 1), dtype='f')

            for i in range(int(self.images.shape[1] / pixel)):
                for j in range(int(self.images.shape[2] / pixel)):
                    img = np.asarray(self.images[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])
                    f, pxx[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = signal.welch(img,
                                                                                                   fs=self.fs,
                                                                                                   axis=0,
                                                                                                   nperseg=segments)

            self.pxx = pxx
            self.f = f

            return

    # noinspection PyArgumentList
    def psd_plot(self, lf_th=1000, mf_th=10000, fname='energycontent.png'):
        # plot previously calculated per pixel psd split up in lf, mf, and hf content

        i = int(np.argwhere(self.f == lf_th))
        j = int(np.argwhere(self.f == mf_th))

        lf = np.sum(self.psd[:i, :, :], axis=0)
        mf = np.sum(self.psd[i:j, :, :], axis=0)
        hf = np.sum(self.psd[j:, :, :], axis=0)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        p1 = ax1.imshow(lf * 100 / lf.max(), cmap='nipy_spectral', vmin=0, vmax=100)
        ax1.set_title('LF Content (<1000Hz)')
        ax2.imshow(mf * 100 / mf.max(), cmap='nipy_spectral', vmin=0, vmax=100)
        ax2.set_title('MF Content (1000Hz - 10000Hz)')
        ax3.imshow(hf * 100 / hf.max(), cmap='nipy_spectral', vmin=0, vmax=100)
        ax3.set_title('HF Content (>10000Hz)')

        cb_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
        fig.colorbar(p1, cax=cb_ax, orientation='horizontal')

        fig.suptitle('Contour shows summed up PSD content, based on 100k images captured at 100k fps, 10deg wedge')
        plt.tight_layout()
        plt.savefig(self.path + fname, dpi=300)

        return

    def coherence(self, reference, segments=2000, pixel=16):
        # calculate coherence between each pixel of image and provided reference signal (e.g Kulite series)
        try:
            import cupy as cp
            import cusignal

        finally:
            gpu = 0

        if gpu == 0:
            cxx_gpu = cp.zeros((int(segments / 2 + 1), self.images.shape[1], self.images.shape[2]), dtype='f')
            f_gpu = cp.zeros(int(segments / 2 + 1), dtype='f')
            ref = cp.tile(reference[:len(self.images.shape), None, None], (1, pixel, pixel))

            for i in range(int(self.images.shape[1] / pixel)):
                for j in range(int(self.images.shape[2] / pixel)):
                    img_gpu = cp.asarray(self.images[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])
                    f_gpu, cxx_gpu[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = cusignal.coherence(
                        img_gpu,
                        ref,
                        fs=self.fs,
                        axis=0,
                        nperseg=segments)

            self.coh = cxx_gpu.get()
            self.f = f_gpu.get()

            return

        else:
            from scipy import signal
            cxx = np.zeros((int(segments / 2 + 1), self.images.shape[1], self.images.shape[2]), dtype='f')
            f = np.zeros(int(segments / 2 + 1), dtype='f')
            ref = np.tile(reference[:len(self.images.shape), None, None], (1, pixel, pixel))

            for i in range(int(self.images.shape[1] / pixel)):
                for j in range(int(self.images.shape[2] / pixel)):
                    img = np.asarray(self.images[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])
                    f, cxx[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = signal.coherence(img,
                                                                                                       ref,
                                                                                                       fs=self.fs,
                                                                                                       axis=0,
                                                                                                       nperseg=segments)

            self.pxx = cxx
            self.f = f

            return

    def coherence_plot(self, lf_th=1000, fname='coherence.png'):
        # plot previously calculated coherence in the low frequency domain
        i = int(np.argwhere(self.f == lf_th))
        lf = np.sum(self.coh[:i, :, :], axis=0)

        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        ax1.imshow(lf / i, cmap='Greys', vmin=-0.05, vmax=0.3)
        fig.suptitle('Contour shows summed up Coherence, based on 100k images captured at 100k fps, 10deg wedge')
        plt.tight_layout()
        plt.savefig(self.path + fname, dpi=300)

        return

    def filter(self, cutoff=2000, order=50, pixel=16, keep_original=1):
        # apply low-pass filter to time series. Will use gpu if applicable
        from scipy import signal
        try:
            import cupy as cp
            import cusignal
        finally:
            gpu = 0

        if keep_original == 1:
            img_filtered = np.zeros(self.images.shape)

        filt = signal.butter(order, cutoff, 'lp', output='sos', fs=self.fs)

        for i in range(int(self.images.shape[1] / pixel)):
            for j in range(int(self.images.shape[2] / pixel)):
                if gpu == 0:
                    img = self.images[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel]
                    filtered = signal.sosfilt(filt, img, axis=0)
                else:
                    img_gpu = cp.asarray(self.images[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])
                    filtered = cusignal.sosfilt(filt, img_gpu, axis=0)
                if keep_original == 1:
                    img_filtered[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = filtered.astype('int16')

                else:
                    self.images[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = filtered.astype('int16')

        return img_filtered if keep_original == 1 else None

    def wiener(self, reference, norm=0, L=2000, keep_original=1):
        # Apply optimal Wiener Filter to image series. Reference needs to be provided. Requires GPU!

        import cupy as cp
        from signal_analysis import wiener_gpu
        if norm == 1:
            reference = reference / reference.max()

        if keep_original == 1:
            img_noise = np.zeros(self.images.shape, dtype='int16')
            img_filtered = np.zeros(self.images.shape, dtype='int16')

        p_ref = cp.asarray(reference, dtype='f')
        p_ref_array = cp.zeros((len(reference) - L, L), dtype='f')
        i = 0
        for n in range(L, len(reference)):
            p_ref_array[i, :] = p_ref[n - L:n][::-1]
            i += 1

        for i in range(self.images.shape[1]):
            for j in range(self.images.shape[2]):
                p_main = cp.asarray(self.images[:, i, j])
                noise, filtered = wiener_gpu(p_main, p_ref, L, p_ref_array)
                if keep_original == 1:
                    img_noise[:, i, j], img_filtered[:, i, j] = noise, filtered
                else:
                    self.images[:, i, j] = noise

            print('Finished %d out of %d pixels' % (i * (self.images.shape[2]) + j + 1,
                                                    self.images.shape[1] * self.images.shape[2]))

        return img_noise, img_filtered if keep_original == 1 else None

    def compare_psd(self, x1, y1, x2, y2, segments=2000):
        # compares the Powerspectra at two different positions of the array
        from signal_analysis import psd as pwelch

        pxx, _, f = pwelch(self.images[:, x1, y1] / (self.images[:, x1, y1].max()), self.fs, segments)
        pxx1, _, f = pwelch(self.images[:, x2, y2] / (self.images[:, x2, y2].max()), self.fs, segments)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 12))
        ax1.semilogx(f, f * pxx)
        ax1.semilogx(f, f * pxx1)

        ax2.loglog(f, pxx)
        ax2.loglog(f, pxx1)
        return
