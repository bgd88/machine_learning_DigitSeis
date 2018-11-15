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

Created on Jul 28, 2016

author: jakeret
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
                           border = 92, sigma = 20, alpha=0.9, width=5):


    image = np.ones((nx, ny, 1))
    label = np.zeros((nx, ny, 3), dtype=np.bool)
    mask = np.zeros((nx, ny), dtype=np.bool)
    for _ in range(cnt):
        a = np.random.randint(border, nx-border)
        b = np.random.randint(border, ny-border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1,100)

        y,x = np.ogrid[-a:nx-a, -b:ny-b]
        m = x*x + y*y <= r*r
        mask = np.logical_or(mask, m)

        image[m] = h

    yStart = np.random.randint(0, 400)
    delY = np.random.randint(290, 310)
    h2 = np.random.randint(250,255)
    mask = np.zeros((nx, ny))
    for tNum in range(3):
        y = yWidth*generate_ts(nx, alpha) + yStart + tNum*delY
        for ii in np.arange(nx):
            for yOff in np.arange(-yWidth, yWidth+1):
                for xOff in np.arange(-xWidth, xWidth+1):
                    xInd = ii + xOff
                    pInd = y[ii] + yOff
                    try:
                        mask[( pInd, xInd)] = 1
                        image[(pInd, xInd)] = h2
                    except:
                        continue


    label[mask, 1] = 1

    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)

    return image, label[..., 1]

    def generate_ts(nx, alpha):
        # np.random.seed(1)
        #x = w = np.random.randint(-high, high, size=nx)
        y = w = np.random.normal(size=nx)
        for t in range(nx):
            y[t] = alpha*y[t-1] + w[t]
        return np.round(y).astype(int)
