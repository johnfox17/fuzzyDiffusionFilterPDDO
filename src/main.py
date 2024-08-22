import sys
import numpy as np
import cv2
import fuzzyDiffusionFilterPDDO
import constants

def main():
    convertToGrayScale = True
    if sys.platform.startswith('linux'):
        pathToReferenceImage = '../data/simData/cameraman.png'
        pathToImage = '../data/simData/noisyCameraman.png' 
                #'../data/simData/cameraman.png'
                #'../data/simData/noisyCameraman.png'
                #'../data/simData/noisyLena.png'
                #'../data/simData/noisyCircle'
                #'../data/simData/noisyLena.png'
                # '../data/simData/Lena.png'
        pathToMembershipFunction = '../data/simData/triangularMembershipFunction.csv'

    else:
        pathToReferenceImage = '..\\data\\simData\\cameraman.png'
        pathToImage = '..\\data\\simData\\noisyCameraman.png' 
            #'..\\data\\simData\\cameraman.png'
            #'..\\data\\simData\\noisyLena.png'
            #'..\\data\\simData\\noisyCircle.png'
            #'..\\data\\simData\\noisyLena.png'
            #'..\\data\\simData\\Lena.png'
        pathToMembershipFunction = '..\\data\\simData\\triangularMembershipFunction.csv'

    image = cv2.imread(pathToImage)
    referenceImage = cv2.imread(pathToReferenceImage)

    if convertToGrayScale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        referenceImage = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('../data/simData/noisyImageGrayScale.jpg', image)
        cv2.imwrite('../data/simData/referenceImageGrayScale.jpg', referenceImage)
        
        
    fuzzyFilter = fuzzyDiffusionFilterPDDO.fuzzyDiffusionFilterPDDO(pathToMembershipFunction)
    fuzzyFilter.solve(referenceImage, image)
    
    print('Done with simulation')


if __name__ == "__main__":
    main()

