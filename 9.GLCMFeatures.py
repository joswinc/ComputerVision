# -*- coding: utf-8 -*-
"""



    - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
    - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1}P_{i,j}|i-j|`
    - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
    - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`
    - 'energy': :math:`\\sqrt{ASM}`
    - 'correlation':
        
"""


from skimage.feature import greycomatrix, greycoprops
from skimage import data


PATCH_SIZE = 21

# open the camera image
image = data.camera()


glcm = greycomatrix(image, [5], [0], 256, symmetric=True, normed=True)

prop = ['contrast', 'dissimilarity', 'homogeneity', 'energy','correlation', 'ASM']
Features=[]
for p in prop:
    Features.append(greycoprops(glcm, p)[0,0])
