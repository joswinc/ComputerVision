# -*- coding: utf-8 -*-
"""


skimage.feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None)[source]
Extract Histogram of Oriented Gradients (HOG) for a given image.

Compute a Histogram of Oriented Gradients (HOG) by

1. (optional) global image normalization

2. computing the gradient image in row and col

3. computing gradient histograms

4. normalizing across blocks

5. flattening into a feature vector

Parameters
image : (M, N[, C]) ndarray. Input image.

orientations : int, optional. Number of orientation bins.

pixels_per_cell : 2-tuple (int, int), optional. Size (in pixels) of a cell.

cells_per_block : 2-tuple (int, int), optional. Number of cells in each block.

block_norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}, optional. Block normalization method:

L1. Normalization using L1-norm.

L1-sqrt. Normalization using L1-norm, followed by square root.

L2. Normalization using L2-norm.

L2-Hys. Normalization using L2-norm, followed by limiting the maximum values to 0.2 (Hys stands for hysteresis) and renormalization using L2-norm. (default) For details, see [R170], [R171].

visualize : bool, optional. Also return an image of the HOG. For each cell and orientation bin, the image contains a line segment that is centered at the cell center, is perpendicular to the midpoint of the range of angles spanned by the orientation bin, and has intensity proportional to the corresponding histogram value.

transform_sqrt : bool, optional. Apply power law compression to normalize the image before processing. DO NOT use this if the image contains negative values. Also see notes section below.

feature_vector : bool, optional. Return the data as a feature vector by calling .ravel() on the result just before returning.

multichannel : boolean, optional. If True, the last image dimension is considered as a color channel, otherwise as spatial.

Returns
out : (n_blocks_row, n_blocks_col, n_cells_row, n_cells_col, n_orient) ndarray

HOG descriptor for the image. If feature_vector is True, a 1D (flattened) array is returned.

hog_image : (M, N) ndarray, optional

A visualisation of the HOG image. Only provided if visualize is True.

"""








import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure


image = data.astronaut()

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True, feature_vector=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()