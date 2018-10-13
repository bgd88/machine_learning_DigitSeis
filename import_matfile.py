#!/usr/bin/env python3
import h5py
import glob
import pandas as pd
from myPath import dataDir
import matplotlib.pyplot as plt
import numpy as np

# Get the path to all .mat files
filenames = glob.glob(dataDir + "images/*.mat")

# Lets do a simple Exmple with the first file in the List
testFile = filenames[0]
f = h5py.File(testFile, 'r')

# Cerate Pandas Dataframe
df = pd.DataFrame()

# Convert from a view to a list to make it simplier/less abstract
keys = list(f.keys())
print("The keys are {}".format(keys))

# Loop over all values
for key in f.keys():
    print(key)
    print(f[key].value)

# Example of extracting variables - They are wrapped in lists for some reason
imInt = f['imInt'].value.astype('int')[0]
pixID = f['pixID'].value.astype('int')[0]
nc = f['nc'].value.astype('int')[0][0]
nr = f['nr'].value.astype('int')[0][0]

plt.style.use('ggplot')

# Note that you need to take care here, since Matlab writes arrays in a Fortran-like
# sequencing whereas Pythin uses a C-like sequencing. They raster differently,
# row then column versus column then row. i.e. Switch nc/nr or specifically tell it
# to use a different ordering

plt.imshow(np.reshape(imInt, [nr, nc], order="F"), cmap="bone")
plt.title("Pixel Intensity Value", fontsize=20)
plt.show()

plt.imshow(np.reshape(pixID, [nr, nc], order="F"), cmap="rainbow")
plt.title("Pixel Label", fontsize=20)
plt.show()
