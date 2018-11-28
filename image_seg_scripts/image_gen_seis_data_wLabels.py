# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Toy example, generates images at random that can be used for training

Created on Nov. 15, 2018

author: bgd88
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import glob
import h5py
import matplotlib.pyplot as plt
from tf_unet.image_util import BaseDataProvider
from myPath import dataDir
from count_crossings import return_images_withPixelCrossings

class DigitSeisDataProvider(BaseDataProvider):
    channels = 1
    n_class = 4

    def __init__(self, nx, ny, binaryImage=True, numFiles=None, mostCrossings=False, **kwargs):
        super(DigitSeisDataProvider, self).__init__()
        if numFiles is not None:
            if mostCrossings:
                self.filenames = return_images_withPixelCrossings(numFiles)
                print("Loaded {} files with the most pixel crossings.".format(numFiles))
            else:
                self.filenames = np.random.choice(glob.glob(dataDir + "images/*.mat"), numFiles)
        else:
            self.filenames = glob.glob(dataDir + "images/*.mat")
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        self.filesUsed =  []

    def _next_data(self):
        fn = np.random.choice(self.filenames)
        self.filesUsed.append(fn.split('/')[-1])
        return create_image_and_label(fn , self.nx, self.ny, **self.kwargs)

def create_image_and_label(filename, nx, ny):

    labels = np.zeros((ny, nx, 4), dtype=np.bool)
    image, mask = read_mat_file(filename, nx, ny)

    # Pixel Codings
    # -1 == Background
    #  0 == Trace
    #  1 == Time Mark
    #  2 == Crossing
    for ii, pixCode in enumerate([-1, 0, 1, 2]):
        labels[mask==pixCode, ii] = 1
    image -= np.amin(image)
    image /= np.amax(image)
    return image, labels

def read_mat_file(filename, nx, ny):
    f = h5py.File(filename, 'r')

    # Example of extracting variables - They are wrapped in lists for some reason
    imInt = f['imInt'].value.astype('int')[0]
    pixID = f['pixID'].value.astype('int')[0]
    nc = f['nc'].value.astype('int')[0][0]
    nr = f['nr'].value.astype('int')[0][0]
    image = np.ones((nr, nc, 1))

    image = np.reshape(imInt, [nr, nc, 1], order="F").astype(float)
    mask = np.reshape(pixID, [nr, nc], order="F")
    if nx < nc:
        maxNX = nc - nx - 1
        xLow  = np.random.randint(0, maxNX)
        xHigh = xLow + nx
        image = image[:,xLow:xHigh,:]
        mask = mask[:,xLow:xHigh]
    f.close()
    return image, mask

def plot_image_label(image, label, pixCode):
    fig, ax = plt.subplots(1,2, figsize=[15,5])
    ax[0].imshow(image[0,:,:, 0], cmap='bone')
    ax[0].set_title('Test Image', fontsize=20)
    ax[1].imshow(label[0,:, :, pixCode])
    ax[1].set_title('True Label', fontsize=20)
    plt.savefig('Test_Image_Labels.pdf', dpi=100, fmt='pdf')
    plt.close()

if __name__ == '__main__':
    generator = DigitSeisDataProvider(2000, 1000)
    image, label = generator(1)
    plot_image_label(image, label, 3)
