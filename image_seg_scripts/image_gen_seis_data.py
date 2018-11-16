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
from tf_unet.image_util import BaseDataProvider

class GrayScaleDataProvider(BaseDataProvider):
    channels = 1
    n_class = 2

    def __init__(self, nx, ny, **kwargs):
        super(GrayScaleDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3

    def _next_data(self):
        return create_image_and_label(self.nx, self.ny, **self.kwargs)

def create_image_and_label(nx,ny, cnt = 10, r_min = 5, r_max = 50,
                           border = 92, sigma = 20, alpha=0.9, xWidth=3, yWidth=20):


    image = np.ones((ny, nx, 1))
    label = np.zeros((ny, nx, 3), dtype=np.bool)
    for _ in range(cnt):
        a = np.random.randint(border, ny-border)
        b = np.random.randint(border, nx-border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1,100)

        x,y = np.ogrid[-a:ny-a, -b:nx-b]
        m = x*x + y*y <= r*r
        # mask = np.logical_or(mask, m)
        image[m] = h

    yStart = np.random.randint(0, 400)
    delY = np.random.randint(290, 310)
    h2 = np.random.randint(250,255)
    mask = np.zeros((ny, nx))
    for tNum in range(3):
        y = yWidth*generate_ts(nx, alpha) + yStart + tNum*delY
        for ii in np.arange(nx):
        #     for yOff in np.arange(-yWidth, yWidth+1):
        #         for xOff in np.arange(-xWidth, xWidth+1):
                    # xInd = ii + xOff
                    # pInd = y[ii] + yOff
                    xMin = max([0, ii-xWidth])
                    xMax = min([nx, ii+xWidth])
                    yMin = max([0, y[ii]-yWidth])
                    yMax = min([ny, y[ii]+yWidth])
                    try:
                        mask[yMin:yMax, xMin:xMax] = 1
                        image[yMin:yMax, xMin:xMax] = h2
                    except:
                        continue

    label[pixID>=0, 1] = 1

    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)
    return image, label[..., 1]

def read_mat_file(filename):
    f = h5py.File(filename, 'r')

    # Example of extracting variables - They are wrapped in lists for some reason
    imInt = f['imInt'].value.astype('int')[0]
    pixID = f['pixID'].value.astype('int')[0]
    nc = f['nc'].value.astype('int')[0][0]
    nr = f['nr'].value.astype('int')[0][0]
    image = np.ones((nr, nc, 1))
    label = np.zeros((nr, nc, 3), dtype=np.bool)

    image = np.reshape(imInt, [nr, nc, 1], order="F")
    pixID = np.reshape(pixID, [nr, nc], order="F")
plt.imshow(, cmap="bone")
plt.title("Pixel Intensity Value", fontsize=20)
plt.show()

plt.imshow(, cmap="rainbow")

def plot_image_label(image, label):
    fig, ax = plt.subplots(1,2, figsize=[15,5])
    ax[0].imshow(image[:, :, 0], cmap='bone')
    ax[0].set_title('Test Image', fontsize=20)
    ax[1].imshow(label[:, :, 1])
    ax[1].set_title('True Label', fontsize=20)
    plt.savefig('Test_Image_Labels.pdf', dpi=100, fmt='pdf')
    plt.close()
