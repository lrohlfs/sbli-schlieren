# Data Analysis Functions
import time
import numpy as np
from scipy import signal, stats
from scipy.linalg import solve_toeplitz

try:
    import cupy as cp
    import cusignal
    # import cupyx.scipy as sp
finally:
    pass


def psd(p_main, f_s, segments):
    # filtered = signal.sosfilt(filt,test[:,i,j])

    f, pxx = signal.welch(p_main, f_s, nperseg=segments, scaling='density')
    f_max = (f[(f * pxx).argmax()])
    return pxx, f, f_max


def cohere(p_main, p_ref, f_s, segments):
    f, pxx = signal.coherence(p_main, p_ref, fs=f_s, nperseg=segments)
    return f, pxx


def pearson(p_main, p_ref):
    r, p = stats.pearsonr(p_main, p_ref)
    return r, p


def wiener(p_main, p_ref, L):
    # t0 = time.time()
    Ns = p_main.size

    g_cc = signal.correlate(p_main, p_ref)
    g = g_cc[Ns - 1:Ns + L - 1]

    r_ac = signal.correlate(p_ref, p_ref)
    r = r_ac[Ns - 1:Ns + L - 1]

    # RR = toeplitz(r.T)

    f = solve_toeplitz(r, g)  # np.linalg.solve(RR, g)

    p_noise = np.zeros(Ns)

    # t1 = time.time()
    # print('Done1 in%6.2f secs' % (t1 - t0))
    # t0 = time.time()

    def calc(p_noise, f, p_ref, L, Ns):
        for n in range(L, Ns, 1):
            p_noise[n - 1] = np.dot(f, p_ref[n - L:n][::-1])  # p_sum
        return p_noise

    p_noise = calc(p_noise, f, p_ref, L, Ns)

    p_filtered = np.subtract(p_main, p_noise)

    # t1 = time.time()
    # print('Done2in%6.2f secs' % (t1 - t0))
    return p_filtered[L:], p_noise[L:]


def wiener_gpu(p_main, p_ref, L, p_ref_array):
    # t0 = time.time()
    p_ref = p_ref * p_main.max()
    Ns = p_main.size

    g_cc = cusignal.correlate(p_main, p_ref)
    g = g_cc[Ns - 1:Ns + L - 1]

    r_ac = cusignal.correlate(p_ref, p_ref)
    r = r_ac[Ns - 1:Ns + L - 1]
    # t1 = time.time()
    # print('Done2in%6.2f secs' % (t1 - t0))
    # RR = sp.linalg.toeplitz(r.T)
    # f = np.linalg.solve(RR.get(),g.get())
    f = solve_toeplitz(r.get(), g.get()).astype('f')
    try:
        del r, r_ac, g, g_cc, p_ref
        cp._default_memory_pool.free_all_blocks()
    except:
        pass

    # p_noise = np.zeros(Ns)

    # p_ref = p_ref.get()

    def calc(p_noise, f, p_ref, L, Ns):
        for n in range(L, Ns, 1):
            p_noise[n - 1] = np.dot(f, p_ref[n - L:n][::-1])  # p_sum
        return p_noise

    p_noise = cp.dot(cp.asarray(f), p_main.max() * p_ref_array_cp.T).astype('f')
    # p_noise = calc(p_noise, f, p_ref.get(), L, Ns)

    p_filtered = cp.subtract(p_main[L:], p_noise)
    # t1 = time.time()
    # print('Done2in%6.2f secs' % (t1 - t0))
    return p_noise.get(), p_filtered.get()

# def wiener_gpu_3D(p_main, p_ref, L):
#     Ns = p_main.shape[0]
#
#     g_cc = cusignal.correlate(p_main, p_ref)
#     g = g_cc[Ns - 1:Ns + L - 1]
#
#     r_ac = cusignal.correlate(p_ref, p_ref)
#     r = r_ac[Ns - 1:Ns + L - 1]
#
#     RR = sp.linalg.toeplitz(r.T)
#     f = np.linalg.solve(RR.get(), g.get())
#     p_noise = np.zeros(Ns)
#
#     p_ref = p_ref.get()
#
#     def calc(p_noise, f, p_ref, L, Ns):
#         for n in range(L, Ns, 1):
#             p_noise[n - 1] = np.dot(f, p_ref[n - L:n][::-1])  # p_sum
#         return p_noise
#
#     p_noise = calc(p_noise, f, p_ref, L, Ns)
#
#     # p_filtered = np.subtract(p_main.get(), p_noise)
#
#     return p_noise[L:]
