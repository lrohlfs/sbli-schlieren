Ns = p_main.shape[0]

g_cc = cusignal.correlate(p_main, p_ref,mode='same')
g_cc1 = cusignal.correlate(p_main[:,0,0], p_ref[:,0,0])

r_ac = cusignal.correlate(p_ref, p_ref,mode='same')
r = r_ac[Ns/2:Ns/2+L,:,:]


g = g_cc[Ns/2:Ns/2+L,:,:]
g1 = g_cc1[Ns - 1:Ns + L - 1]


plt.plot(g[:,0,0].get())
plt.plot(g1.get())

del export
np.save(path+'lp_filtered_fluc.npy',[img_f,p_main.get(),p_ref.get()])

def wiener_gpu_3D(p_main, p_ref, L):

    Ns = p_main.shape[0]

    g_cc = cusignal.correlate(p_main, p_ref)
    g = g_cc[Ns - 1:Ns + L - 1]

    r_ac = cusignal.correlate(p_ref, p_ref)
    r = r_ac[Ns - 1:Ns + L - 1]

    RR = sp.linalg.toeplitz(r.T)
    f = np.linalg.solve(RR.get(),g.get())
    p_noise = np.zeros(Ns)

    p_ref = p_ref.get()

    def calc(p_noise, f, p_ref, L, Ns):
        for n in range(L, Ns, 1):
            p_noise[n - 1] = np.dot(f, p_ref[n - L:n][::-1])  # p_sum
        return p_noise

    p_noise = calc(p_noise, f, p_ref, L, Ns)

    # p_filtered = np.subtract(p_main.get(), p_noise)

    return p_noise[L:]
