# Only run Calls to other functions from here!
from schlieren import Schlieren
from kulite import Kulite
from funtions import coherence_line

# Path with either npy file or subfolder with images
path = 'D:/Arbeit_Homeoffice/Forschung/Schlieren/'
path = '/media/lennart/a6bd1732-be49-4912-9ae5-51db4e876215/home/lennart/Data/sg06/'

# Initialize Classes
schlieren = Schlieren(path)
# kulite = Kulite(path)

# Load Schlieren Images
# Names: SG06_20000_settinvariation_4.cine, 40000_Test.cine
schlieren.load('SG06_20000_settinvariation_3.cine', fs=20000)
# kulite.load('kulite_series.npy', end=10000)


# Calculate Coherence to reference point in image as specified by x and y
# x = 160
# y_range = np.linspace(20, 420, 21).astype('uint16')
# coherence_line(schlieren, x, y_range)
# schlieren.coherence_plot(fname='coherence_bubble_2')

# Calculate and Plot Energy Spectrum
schlieren.psd(pixel=16,segments=1000)
schlieren.psd_plot(fname='test.png')
