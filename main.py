# Only run Calls to other functions from here!
from schlieren import Schlieren
from kulite import Kulite

# Path with either npy file or subfolder with images
path = 'D:/Arbeit_Homeoffice/Forschung/Schlieren/'

# Initialize Classes
schlieren = Schlieren(path)
kulite = Kulite(path)

# Load Schlieren Images

schlieren.load('filename.npy', end=10000)
kulite.load('kulite_series.npy', end=10000)