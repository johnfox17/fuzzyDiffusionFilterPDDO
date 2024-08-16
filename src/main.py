import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import fuzzyDiffusionFilterPDDO
import constants
#from multiprocessing import Process
#from multiprocessing import Pool, TimeoutError

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
    #p0 = Process(target=fuzzyFilter.solve, args=(image[:,:,0],0,))
    #p0.start()
    #p0.join()

    #p1 = Process(target=fuzzyFilter.solve, args=(image[:,:,1],1,))
    #p1.start()
    #p1.join()

    #p2 = Process(target=fuzzyFilter.solve, args=(image[:,:,2],2,))
    #p2.start()
    #p2.join()
    
    #p0.join()
    #p1.join()
    #p2.join()
    print('Done')
    #print(np.shape(image[:,:,0]))
    #fuzzyFilter.solve(image)





    #imagePlot = plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    #plt.show()
    #signals = createSignals.createSignals()
    #signals.solve()
    

    #np.savetxt('..\\data\\localSmoothness.csv',  fuzzyFilter.localSmoothness, delimiter=",")
    #np.savetxt('..\\data\\generalAverage.csv',  fuzzyFilter.generalAverage, delimiter=",")
    #np.savetxt('..\\data\\denoisedImages.csv',  fuzzyFilter.denoisedImages, delimiter=",")
    #np.savetxt('..\\data\\neighboringPixels.csv',  fuzzyFilter.neighboringPixels, delimiter=",")
    #print(fuzzyFilter.neighboringPixels)
    #np.savetxt('/home/doctajfox/Documents/Thesis/1DPDDOKernels/data/xi2_2.csv', PDDOKernel2_2.xis, delimiter=",")


if __name__ == "__main__":
    main()

