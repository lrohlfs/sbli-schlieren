## Data Analysis Functions
import time
import numpy as np
from scipy import signal, stats
from scipy.linalg import toeplitz


def psd(p_main, f_s, segments):
    # filtered = signal.sosfilt(filt,test[:,i,j])

    f, pxx = signal.welch(p_main, f_s, nperseg=segments, scaling='density')
    f_max = (f[(f * pxx).argmax()])
    return pxx, f_max, f


def cohere(p_main, p_ref, f_s, segments):
    f, pxx = signal.coherence(p_main, p_ref, fs=f_s, nperseg=segments)
    return f, pxx


def pearson(p_main, p_ref):
    r, p = stats.pearsonr(p_main, p_ref)
    return r, p


def wiener(p_main,p_ref,L):
    t0 = time.time()
    Ns = p_main.size

    g_cc = signal.correlate(p_main, p_ref)
    g = g_cc[Ns - 1:Ns + L - 1]

    r_ac = signal.correlate(p_ref, p_ref)
    r = r_ac[Ns - 1:Ns + L - 1]

    RR = toeplitz(r.T)


    f = np.linalg.solve(RR,g)

    p_noise = np.zeros(Ns)

    def calc(p_noise, f, p_ref, L, Ns):
        for n in range(L, Ns, 1):
            p_noise[n-1] = np.dot(f,p_ref[n - L:n][::-1])  # p_sum
        return p_noise

    p_noise = calc(p_noise, f, p_ref, L, Ns)


    p_filtered = np.subtract(p_main, p_noise)

    t1 = time.time()
    print('Done in%6.2f secs' % (t1 - t0))
    return p_filtered[L:], p_noise[L:]
