import numpy as np
from skimage import exposure
import cv2

image = np.loadtxt('../data/outputPDDODerivative3/130_denoisedImage.csv')

#quantize image
#image = np.divide(image,2**8)
#Contrast stretching
p0, p1 = np.percentile(image, (20, 99.9))

image = np.multiply(255,exposure.rescale_intensity(image, in_range=(p0, p1)))

#np.savetxt('../data/output_0.8Mean2/rescaledImage/rescaledImage.csv', image)
cv2.imwrite('../data/outputPDDODerivative3/rescaledImage/rescaledImage.jpg', image)
