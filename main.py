# Only run Calls to other functions from here!
from schlieren import Schlieren
from kulite import Kulite
from funtions import coherence_line

from matplotlib import pyplot as plt
from scipy import signal

# Path with either npy file or subfolder with images
path = 'D:/Arbeit_Homeoffice/Forschung/Schlieren/'
path = '/home/lennart/Data/Sg10/'
path = 'E:/TSK_2021/Exports/'
# Initialize Classes
schlieren = Schlieren(path)
# kulite = Kulite(path)

# Load Schlieren Images
# Names: SG06_20000_settinvariation_4.cine, 40000_Test.cine
schlieren.load('SG10PLS/', fs=5000)
# kulite.load('kulite_series.npy', end=10000)


# Calculate Coherence to reference point in image as specified by x and y
x = 160
y_range = np.linspace(20, 420, 21).astype('uint16')
coherence_line(schlieren, x, y_range)
# schlieren.coherence_plot(fname='coherence_bubble_2')

# Calculate and Plot Energy Spectrum
# schlieren.psd(pixel=32,segments=1000)
# schlieren.psd_plot()

edges = schlieren.canny_edge(par=(3, 30, 60))
shockpos = []
for edge in edges:
    shockpos.append(np.argwhere(edge[180,:]==True)[0][0])
f,pxx = signal.welch(shockpos,fs=5000,nperseg = 500)
plt.semilogx(f,f*pxx)


fig, ax1 = plt.subplots(figsize=(12, 5))
i = 0
for edge in edges:
    ax1.imshow(edge[:250, :], cmap='Greys_r')
    plt.tight_layout()
    plt.savefig(path + 'edges/img_' + format(i, '03d') + '.png')
    ax1.cla()
    print(i)
    i = i + 1


def create_gif(path, sname='test.gif', duration=0.1):
    import imageio
    files = os.listdir(path)
    imgs = []
    for file in files:
        imgs.append(imageio.imread(path + file))
    imageio.mimsave(path + sname, imgs, duration=duration)


create_gif(path + 'edges/')

plt.imshow(edges[5])
plt.show()
