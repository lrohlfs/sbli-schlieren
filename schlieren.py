import os
import h5py

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from pims import Cine
from datetime import date


class Schlieren(object):
    images: ndarray

    def __init__(self, path, h5file='Results.h5', h5grp='Dez_2021/Control/Schlieren'):

        # Definitions
        self.fname = None
        self.path = path
        self.fs = 100000

        self.h5file = self.path + h5file
        self.h5grp = h5grp
        self.h5 = h5py.File(self.h5file, 'a')

        self.gpu = 1
        self.img_mean = np.array([])
        self.img_size = (0, 0)
        self.pxx = np.array([])
        self.csd = np.array([])
        self.phase = np.array([])
        self.f = np.array([])
        self.coh = np.array([])

        return

    def load(self, name, start=0, end=None, fs=100000, h5save=True):
        print(name)
        self.fname = name
        # Load schlieren images either processed from npy files or from cine or loose files in folder.
        if name.endswith('npy'):
            try:
                self.images = np.load(self.path + name)
            except:
                return print('%s could not be found. Check Path' % name)

        elif name.endswith('/'):
            try:
                files = sorted(os.listdir(self.path + name))[start:end]
                self.images = np.array([plt.imread(self.path + name + file) for file in files], dtype='uint8')
                self.img_mean = np.mean(self.images, axis=0, dtype='int16')
                # self.images = np.subtract(self.images, self.img_mean)
            except:
                return print('%s could not be found. Check Path' % name)

        elif name.endswith('cine'):
            try:
                print('Cine detected')
                raw = Cine(self.path + name)[start:end]
                self.images = np.array(raw, dtype='uint8')
                self.img_mean = np.mean(self.images, axis=0).astype('int16')
            except:
                print('something went wrong')
            # finally:
            #     return print('%s could not be found. Check Path' % name)


        else:
            print('Filetype unknown. Check if method is supported')
            return
        self.img_size = self.img_mean.shape
        max_size = Cine(self.path + name).shape
        if h5save:
            try:
                grp = self.h5.create_group(self.h5grp)
            except:
                grp = self.h5[self.h5grp]

            self.savetoh5(self.images, 'img_raw')
            self.savetoh5(self.img_mean, 'img_mean')

            grp.attrs['Filename'] = name
            grp.attrs['FPS'] = fs
            grp.attrs['Start-End'] = [start, end]

        self.fs = fs
        return

    def load_h5(self):
        self.images = self.h5[self.h5grp + '/img_raw']
        self.img_mean = self.h5[self.h5grp + '/img_mean']
        try:
            self.img_fluc = self.h5[self.h5grp + '/img_fluc']
        except:
            pass

        self.fs = self.h5[self.h5grp].attrs['FPS']
        self.fname = self.h5[self.h5grp].attrs['Filename']
        print(self.h5[self.h5grp].attrs['Start-End'])
        return

    def savetoh5(self, dset, dname):

        try:
            grp = self.h5.create_group(self.h5grp)
        except:
            grp = self.h5[self.h5grp]

        try:
            data = grp.create_dataset(dname, data=dset)
        except:
            del grp[dname]
            data = grp.create_dataset(dname, data=dset)
        data.attrs['Size'] = dset.shape
        data.attrs['Last Modified'] = str(date.today()).replace("-", "_")
        return

    def loadfromh5(self, dname):
        grp = self.h5[self.h5grp]
        data = grp[dname]
        return data

    def calc_fluc(self, h5save=True):
        self.img_fluc = np.subtract(self.images, self.img_mean)
        if h5save:
            grp = self.h5[self.h5grp]
            try:
                fluc = grp.create_dataset('img_fluc', data=self.img_fluc)
            except:
                del grp['img_fluc']
                fluc = grp.create_dataset('img_fluc', data=self.img_fluc)
                fluc.attrs['Size'] = self.img_fluc.shape
        return

    def psd(self, segments=2000, pixel=16):
        # calculate per pixel psd for entire image. Uses gpu if available
        try:
            import cupy as cp
            import cusignal

        finally:
            gpu = 0

        if gpu == 0:
            pxx_gpu = cp.zeros((int(segments / 2 + 1), self.img_fluc.shape[1], self.img_fluc.shape[2]), dtype='f')
            f_gpu = cp.zeros(int(segments / 2 + 1), dtype='f')

            for i in range(int(self.img_fluc.shape[1] / pixel)):
                for j in range(int(self.img_fluc.shape[2] / pixel)):
                    img_gpu = cp.asarray(self.img_fluc[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])
                    f_gpu, pxx_gpu[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = cusignal.welch(
                        img_gpu,
                        fs=self.fs,
                        axis=0,
                        nperseg=segments)
            self.pxx = pxx_gpu.get()
            self.f = f_gpu.get()
            del pxx_gpu
            del f_gpu
            del img_gpu

            return

        else:
            from scipy import signal
            pxx = np.zeros((int(segments / 2 + 1), self.img_fluc.shape[1], self.img_fluc.shape[2]), dtype='f')
            f = np.zeros(int(segments / 2 + 1), dtype='f')

            for i in range(int(self.img_fluc.shape[1] / pixel)):
                for j in range(int(self.img_fluc.shape[2] / pixel)):
                    img = np.asarray(self.img_fluc[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])
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

        lf = np.sum(self.pxx[:i, :, :], axis=0)
        mf = np.sum(self.pxx[i:j, :, :], axis=0)
        hf = np.sum(self.pxx[j:, :, :], axis=0)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        p1 = ax1.imshow(lf, cmap='nipy_spectral')
        ax1.set_title('LF Content (<%dHz)' % lf_th)
        fig.colorbar(p1, ax=ax1, shrink=0.75)
        p2 = ax2.imshow(mf, cmap='nipy_spectral')
        ax2.set_title('MF Content (%dHz - %dHz)' % (lf_th, mf_th))
        fig.colorbar(p2, ax=ax2, shrink=0.75)
        p3 = ax3.imshow(hf, cmap='nipy_spectral')
        ax3.set_title('HF Content (>%dHz)' % mf_th)
        fig.colorbar(p3, ax=ax3, shrink=0.75)

        # cb_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
        # fig.colorbar(p1, cax=cb_ax, orientation='horizontal')

        fig.suptitle('Contour shows summed up PSD content')
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
            cxx_gpu = cp.zeros((int(segments / 2 + 1), self.img_fluc.shape[1], self.img_fluc.shape[2]), dtype='f')
            f_gpu = cp.zeros(int(segments / 2 + 1), dtype='f')
            ref = cp.tile(reference[:len(self.img_fluc), None, None], (1, pixel, pixel))

            for i in range(int(self.img_fluc.shape[1] / pixel)):
                for j in range(int(self.img_fluc.shape[2] / pixel)):
                    img_gpu = cp.asarray(self.img_fluc[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])
                    f_gpu, cxx_gpu[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = cusignal.coherence(
                        img_gpu,
                        ref,
                        fs=self.fs,
                        axis=0,
                        nperseg=segments)

            self.coh = cxx_gpu.get()
            self.f = f_gpu.get()
            del cxx_gpu
            del f_gpu
            del img_gpu
            return

        else:
            from scipy import signal
            cxx = np.zeros((int(segments / 2 + 1), self.img_fluc.shape[1], self.img_fluc.shape[2]), dtype='f')
            f = np.zeros(int(segments / 2 + 1), dtype='f')
            ref = np.tile(reference[:len(self.img_fluc.shape), None, None], (1, pixel, pixel))

            for i in range(int(self.img_fluc.shape[1] / pixel)):
                for j in range(int(self.img_fluc.shape[2] / pixel)):
                    img = np.asarray(self.img_fluc[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])
                    f, cxx[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = signal.coherence(img,
                                                                                                       ref,
                                                                                                       fs=self.fs,
                                                                                                       axis=0,
                                                                                                       nperseg=segments)

            self.pxx = cxx
            self.f = f

            return

    def coherence_test(self, reference, segments=2000, pixel=16):
        # calculate coherence between each pixel of image and provided reference signal (e.g Kulite series)
        try:
            import cupy as cp
            import cusignal

        finally:
            gpu = 0

        if gpu == 0:
            cxx_gpu = cp.zeros((int(segments / 2 + 1), self.img_fluc.shape[1], self.img_fluc.shape[2]), dtype='f')
            f_gpu = cp.zeros(int(segments / 2 + 1), dtype='f')
            ref = cp.tile(reference[:len(self.img_fluc), None, None], (1, pixel, pixel))

            for i in range(int(self.img_fluc.shape[1] / pixel)):
                for j in range(int(self.img_fluc.shape[2] / pixel)):
                    img_gpu = cp.asarray(self.img_fluc[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])
                    f_gpu, cxx_gpu[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = cusignal.coherence(
                        img_gpu,
                        ref,
                        fs=self.fs,
                        axis=0,
                        window='hamming',
                        nperseg=segments,
                    )

            self.coh = cxx_gpu.get()
            self.f = f_gpu.get()
            del cxx_gpu
            del f_gpu
            del img_gpu
            return

        else:
            from scipy import signal
            cxx = np.zeros((int(segments / 2 + 1), self.img_fluc.shape[1], self.img_fluc.shape[2]), dtype='f')
            f = np.zeros(int(segments / 2 + 1), dtype='f')
            ref = np.tile(reference[:len(self.img_fluc.shape), None, None], (1, pixel, pixel))

            for i in range(int(self.img_fluc.shape[1] / pixel)):
                for j in range(int(self.img_fluc.shape[2] / pixel)):
                    img = np.asarray(self.img_fluc[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])
                    f, cxx[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = signal.coherence(img,
                                                                                                       ref,
                                                                                                       fs=self.fs,
                                                                                                       axis=0,
                                                                                                       nperseg=segments)

            self.pxx = cxx
            self.f = f

            return

    def spectral_complete(self, reference, segments=2000, pixel=16):
        import cupy as cp
        import cusignal

        csd_gpu = cp.zeros((int(segments / 2 + 1), self.img_fluc.shape[1], self.img_fluc.shape[2]), dtype='complex64')
        pxx_gpu = cp.zeros((int(segments / 2 + 1), self.img_fluc.shape[1], self.img_fluc.shape[2]), dtype='f')
        cxy_gpu = cp.zeros((int(segments / 2 + 1), self.img_fluc.shape[1], self.img_fluc.shape[2]), dtype='f')

        f_gpu = cp.zeros(int(segments / 2 + 1), dtype='f')
        ref = cp.tile(reference[:len(self.img_fluc), None, None], (1, pixel, pixel))

        _, pyy_gpu = cusignal.welch(ref, fs=self.fs, axis=0, nperseg=segments)

        for i in range(int(self.img_fluc.shape[1] / pixel)):
            for j in range(int(self.img_fluc.shape[2] / pixel)):
                img_gpu = cp.asarray(self.img_fluc[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])

                f_gpu, pxx_gpu[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = cusignal.welch(
                    img_gpu,
                    fs=self.fs,
                    axis=0,
                    nperseg=segments)
                # print(pxx_gpu.shape)
                f_gpu, csd_gpu[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = cusignal.csd(
                    img_gpu,
                    ref,
                    fs=self.fs,
                    axis=0,
                    nperseg=segments)
                # print(csd_gpu.dtype)
                cxy_gpu[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = \
                    abs(csd_gpu[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel]) ** 2 / \
                    (pxx_gpu[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] * pyy_gpu)

        self.f = f_gpu.get()
        self.csd = csd_gpu.get()
        self.phase = np.angle(self.csd)

        self.pxx = pxx_gpu.get()
        self.coh = cxy_gpu.get()

        del cxy_gpu
        del csd_gpu
        del pxx_gpu
        del f_gpu
        del img_gpu
        return

    def coherence_plot(self, lf_th=1000, fname='coherence.png'):
        # plot previously calculated coherence in the low frequency domain
        i = int(np.argwhere(self.f == lf_th))
        lf = np.sum(self.coh[:i, :, :], axis=0)

        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        ax1.imshow(lf / i, cmap='Greys', vmin=-0.05, vmax=0.6)
        fig.suptitle('Contour shows summed up Coherence')
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
            img_filtered = np.zeros(self.img_fluc.shape)

        filt = signal.butter(order, cutoff, 'lp', output='sos', fs=self.fs)

        for i in range(int(self.img_fluc.shape[1] / pixel)):
            for j in range(int(self.img_fluc.shape[2] / pixel)):
                if gpu == 0:
                    img = self.img_fluc[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel]
                    filtered = signal.sosfilt(filt, img, axis=0)
                else:
                    img_gpu = cp.asarray(self.img_fluc[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel])
                    filtered = cusignal.sosfilt(filt, img_gpu, axis=0)
                if keep_original == 1:
                    img_filtered[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = filtered.astype('int16')

                else:
                    self.img_fluc[:, i * pixel:(i + 1) * pixel, j * pixel:(j + 1) * pixel] = filtered.astype('int16')

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

        pxx, f, _ = pwelch(self.images[:, x1, y1] / (self.images[:, x1, y1].max()), self.fs, segments)
        pxx1, f, _ = pwelch(self.images[:, x2, y2] / (self.images[:, x2, y2].max()), self.fs, segments)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 12))
        ax1.semilogx(f, f * pxx)
        ax1.semilogx(f, f * pxx1)

        ax2.loglog(f, pxx)
        ax2.loglog(f, pxx1)
        return

    def canny_edge(self, start=0, end=None, par=(2, 50, 100)):
        from skimage import feature
        edges = []
        imgs = self.images[start:end, :, :]
        for img in imgs:
            edges.append(feature.canny(img, sigma=par[0], low_threshold=par[1], high_threshold=par[2]))
        return edges

    def create_phaseaverage(self, f_phase):
        orig_shape = self.images.shape
        n_phases = int((orig_shape[0] / self.fs) * f_phase)
        n_phasepoints = int(self.fs / f_phase)
        self.images = np.asarray(self.images).reshape([n_phases, n_phasepoints, orig_shape[1], orig_shape[2]])
        self.images_pa = np.mean(self.images, axis=0)
        self.savetoh5(self.images_pa, 'images_pa')
        self.h5[self.h5grp].attrs['f_phase'] = f_phase
        return
