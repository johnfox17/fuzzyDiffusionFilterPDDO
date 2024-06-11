import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import fuzzyDiffusionFilterPDDO
import constants

def main():
    convertToGrayScale = False
    for iThreshold in constants.THRESHOLDS:
        if sys.platform.startswith('linux'):
            pathToLena = \
                '../data/simData/noisyLena.png'
                #'../data/simData/noisyCircle'
                #'../data/simData/noisyLena.png'
                # '../data/simData/Lena.png'
            pathToMembershipFunction = '../data/simData/triangularMembershipFunction.csv'

        else:
            pathToLena = \
                '..\\data\\simData\\noisyLena.png'
                #'..\\data\\simData\\noisyCircle.png'
                #'..\\data\\simData\\noisyLena.png'
                #'..\\data\\simData\\Lena.png'
            pathToMembershipFunction = '..\\data\\simData\\triangularMembershipFunction.csv'

        image = cv2.imread(pathToLena)
        
        if convertToGrayScale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        fuzzyFilter = fuzzyDiffusionFilterPDDO.fuzzyDiffusionFilterPDDO(image,pathToMembershipFunction,iThreshold)
        fuzzyFilter.solve()





    imagePlot = plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()
    #signals = createSignals.createSignals()
    #signals.solve()
    

    #np.savetxt('..\\data\\localSmoothness.csv',  fuzzyFilter.localSmoothness, delimiter=",")
    #np.savetxt('..\\data\\generalAverage.csv',  fuzzyFilter.generalAverage, delimiter=",")
    #np.savetxt('..\\data\\denoisedImages.csv',  fuzzyFilter.denoisedImages, delimiter=",")
    #np.savetxt('..\\data\\neighboringPixels.csv',  fuzzyFilter.neighboringPixels, delimiter=",")
    print(fuzzyFilter.neighboringPixels)
    #np.savetxt('/home/doctajfox/Documents/Thesis/1DPDDOKernels/data/xi2_2.csv', PDDOKernel2_2.xis, delimiter=",")


if __name__ == "__main__":
    main()

