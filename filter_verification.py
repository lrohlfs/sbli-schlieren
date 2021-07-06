import numpy as np
from matplotlib import pyplot as plt
import time
from scipy import signal, stats
from scipy.linalg import toeplitz


x = np.linspace(0,2,100000)
p_clean = np.sin(200*x)+0.25*np.cos(1000*x)

noise = np.random.normal(0,1,100000)

p_main = p_clean + 0.2*noise

p_ref = np.cos(27*x) + 0.2*noise

L = 5000

def Wiener_par(p_main,p_ref,L):
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

    t0 = time.time()
    p_noise = calc(p_noise, f, p_ref, L, Ns)
    t1 = time.time()
    print('Done in%6.2f secs' % (t1 - t0))


    p_filtered = np.subtract(p_main, p_noise)

    t1 = time.time()
    print('Done in%6.2f secs' % (t1 - t0))
    return f, p_filtered[L:], p_noise[L:]


f,p_filter,p_noise = Wiener_par(p_main,p_ref,L)

plt.figure(1)
plt.plot(p_noise)
plt.plot(0.2*noise[L:])


plt.figure(2)
plt.plot(p_filter)
plt.plot(p_clean[L:])



