import constants
import numpy as np
from sklearn.neighbors import KDTree
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

class fuzzyDiffusionFilterPDDO:
    def __init__(self, image, pathToMembershipFunction, threshold):
        self.image = image
        self.numChannels = constants.NUMCHANNELS
        self.pathToMembershipFunction = pathToMembershipFunction 
        self.l1 = constants.L1
        self.l2 = constants.L2
        self.Nx = constants.NX
        self.Ny = constants.NY
        self.dx = self.l1/self.Nx
        self.dy = self.l2/self.Ny
        self.horizon = constants.HORIZON
        self.Dn = constants.Dn
        self.q = constants.q
        self.finalTime = constants.FINALTIME
        self.dt = constants.DELTAT
        self.lambd = constants.LAMBDA
        self.GMask = constants.GMASK
        self.gCenter = constants.GCENTER
        self.threshold = threshold
        self.gGradient = constants.gGradient
        self.g10 = constants.g10
        self.g01 = constants.g01
        self.dindexoffset = constants.DINDEXOFFSET
        self.kerneldim = constants.KERNELDIM
        self.rhsMask = constants.RHSMASK

    def createPDDOKernelMesh(self):
        indexing = 'xy'
        xCoords = np.arange(self.dx/2, self.Nx*self.dx, self.dx)
        yCoords = np.arange(self.dy/2, self.Ny*self.dy, self.dy)
        xCoords, yCoords = np.meshgrid(xCoords, yCoords, indexing=indexing)
        xCoords = xCoords.reshape(-1, 1)
        yCoords = yCoords.reshape(-1, 1)
        self.coordinateMesh = np.array([xCoords[:,0], yCoords[:,0]]).T 

    def findNeighboringPixels(self):
        tree = KDTree(self.coordinateMesh, leaf_size=2)
        neighboringPixels = tree.query_radius(self.coordinateMesh, r = self.dx*self.horizon)
        self.neighboringPixels = neighboringPixels.reshape((self.Nx,self.Ny))
    
    def addBoundary(self):
        self.Nx = self.Nx + 2*int(self.horizon)
        self.Ny = self.Ny + 2*int(self.horizon)
        image = np.zeros((self.Nx, self.Ny, self.numChannels))
        for iChan in range(self.numChannels):
            image[:,:,iChan] = np.pad(self.image[:,:,iChan],int(self.horizon),mode='symmetric')
        self.image = image

    def loadMembershipFunction(self):
        self.membershipFunction = np.loadtxt(self.pathToMembershipFunction, delimiter=",")
        
    def loadGradientMembershipFunctions(self):
        self.D_neg6 = constants.D_NEG6
        self.D_neg5 = constants.D_NEG5
        self.D_neg4 = constants.D_NEG4
        self.D_neg3 = constants.D_NEG3
        self.D_neg2 = constants.D_NEG2
        self.D_neg1 = constants.D_NEG1
        self.D_0 = constants.D_0
        self.D_1 = constants.D_1
        self.D_2 = constants.D_2
        self.D_3 = constants.D_3
        self.D_4 = constants.D_4
        self.D_5 = constants.D_5
        self.D_6 = constants.D_6

    def assignMembership(self):
        pixelMemberships = []
        for iChan in range(self.numChannels):
            pixelMembershipsChan = []
            for iCol in range(self.Nx):
                for iRow in range(self.Ny):
                    currentPixelMembership = []
                    currentPixelMembership.append(self.image[iCol,iRow,iChan])
                    membershipIndex = int(self.image[iCol,iRow,iChan])
                    currentPixelMembership.append(list(self.membershipFunction[membershipIndex])[0])
                    currentPixelMembership.append(list(self.membershipFunction[membershipIndex])[1])
                    pixelMembershipsChan.append(currentPixelMembership)
            pixelMemberships.append(pixelMembershipsChan)
        pixelMemberships = np.array(pixelMemberships)
        self.pixelMemberships = pixelMemberships

    def createFuzzyMembershipImage(self):
        fuzzyMembershipImage = []
        for iChan in range(self.numChannels):
            fuzzyMembershipImage.append(self.pixelMemberships[iChan,:,1].reshape((self.Nx,self.Ny)))
        self.fuzzyMembershipImage = np.array(fuzzyMembershipImage)

    def findFuzzyDerivativeRule(self):
        D10 = []
        D01 = []
        for iChan in range(self.numChannels):
            DChannel10 = []
            DChannel01 = []
            for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
                for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                    DChannel10.append(np.sum(np.multiply(self.g10,self.fuzzyMembershipImage[iChan, iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).flatten()).astype(int))
                    DChannel01.append(np.sum(np.multiply(self.g01,self.fuzzyMembershipImage[iChan, iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).flatten()).astype(int))
            D10.append(np.pad(np.array(DChannel10).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))),int(self.horizon),mode='symmetric'))
            D01.append(np.pad(np.array(DChannel01).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))),int(self.horizon),mode='symmetric'))

        self.D10 = D10
        self.D01 = D01
        
    def calculateGradient(self):
        gradient10 = []
        gradient01 = []
        for iChan in range(self.numChannels):
            gradientChannel10 = []
            gradientChannel01 = []
            for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
                for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                    D10 = np.multiply(self.GMask,self.D10[iChan][iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int).flatten()
                    D01 = np.multiply(self.GMask,self.D01[iChan][iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int).flatten()
                    L = np.multiply(self.GMask,self.image[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1, iChan]).astype(int).flatten()
                    muPrem = np.sum(self.membershipFunction[L,1])#This might have to be changed to include all channels of image
                    gCents10 = []
                    for iD in D10:
                        gCents10.append(self.gCenter[np.abs(self.gCenter-iD).argmin()])
                    gCents10 = np.array(gCents10).reshape((self.kerneldim,self.kerneldim))
                    gradientChannel10.append(np.sum(np.multiply(self.GMask,(np.multiply(gCents10, self.membershipFunction[L,1].reshape((self.kerneldim,self.kerneldim))))/muPrem).flatten()))
                    gCents01 = []
                    for iD in D01:
                        gCents01.append(self.gCenter[np.abs(self.gCenter-iD).argmin()])
                    gCents01 = np.array(gCents01).reshape((self.kerneldim,self.kerneldim))
                    gradientChannel01.append(np.sum(np.multiply(self.GMask,(np.multiply(gCents01, self.membershipFunction[L,1].reshape((self.kerneldim,self.kerneldim))))/muPrem).flatten()))
            gradient10.append(np.array(gradientChannel10).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))))
            gradient01.append(np.array(gradientChannel01).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))))
        
        #gradient = np.array(np.sum(gradient,axis=0))
        self.gradient10 = np.array(gradient10)
        self.gradient01 = np.array(gradient01)
        #self.gradient10 = np.divide(gradient10, np.max(np.abs(gradient10)))
        #self.gradient01 = np.divide(gradient01, np.max(np.abs(gradient01)))
        self.gradient = np.divide(self.gradient10 + self.gradient01,np.linalg.norm(self.gradient10 + self.gradient01, axis=0))
        print(np.shape(self.gradient))
        a = input('').split(" ")[0]
    def calculateCoefficients(self):
        coefficients = [] 
        #gradient = np.pad(self.gradient,2*int(self.horizon),mode='symmetric')
        for iChan in range(self.numChannels):
            coefficientsChannel = []
            #gradient10 = np.pad(self.gradient10[iChan],2*int(self.horizon),mode='symmetric')
            #gradient01 = np.pad(self.gradient01[iChan],2*int(self.horizon),mode='symmetric')
            gradient10 = self.gradient10[iChan]
            gradient01 = self.gradient01[iChan]
            for iCol in range(self.Nx-int(2*self.horizon)):
                for iRow in range(self.Ny-int(2*self.horizon)):
                    iGradientMagnitude = np.sqrt(gradient10[iRow,iCol]**2 + gradient01[iRow,iCol]**2)
                    K = 0.8
                    coefficientsChannel.append(np.exp(-np.power(np.abs(np.divide(iGradientMagnitude,K)),2)))
            coefficients.append(np.transpose(np.array(coefficientsChannel).reshape((self.Nx-int(2*self.horizon), self.Ny-int(2*self.horizon)))))
        self.coefficients = np.array(coefficients)
        #print(np.shape(self.coefficients))
        #a = input('').split(" ")[0]
    
    def solveRHS(self):
        RHS = []
        for iChan in range(self.numChannels):
            gradientChannel = np.pad(self.gradient10[iChan]+self.gradient01[iChan],int(self.horizon),mode='symmetric')
            coefficients = np.pad(self.coefficients[iChan],int(self.horizon), mode='symmetric') 
            RHSChannel = []
            for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
                for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                    RHSChannel.append(np.sum(np.multiply(coefficients[iRow-int(self.horizon-1):iRow+int(self.horizon),iCol-int(self.horizon-1):iCol+int(self.horizon)], np.multiply(self.rhsMask,gradientChannel[iRow-int(self.horizon-1):iRow+int(self.horizon),iCol-int(self.horizon-1):iCol+int(self.horizon)]))))
                    #RHSChannel.append(np.sum(np.multiply(coefficients[iRow,iCol], np.multiply(self.GMask,gradientChannel[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]))))
            RHS.append(np.transpose(np.array(RHSChannel).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon)))))
        self.RHS = np.array(RHS) 

    def checkSaturation(self):       
        self.Nx = self.Nx - 2*int(self.horizon)
        self.Ny = self.Ny - 2*int(self.horizon)
        denoisedImageChannel0 = self.denoisedImage[0].flatten()
        denoisedImageChannel1 = self.denoisedImage[1].flatten()
        denoisedImageChannel2 = self.denoisedImage[2].flatten()
        if (np.max(np.absolute(denoisedImageChannel0))>255 or  np.max(np.absolute(denoisedImageChannel1))>255 or np.max(np.absolute(denoisedImageChannel2))>255):
            image = []
            while (np.max(np.absolute(denoisedImageChannel0))>255 or  np.max(np.absolute(denoisedImageChannel1))>255 or np.max(np.absolute(denoisedImageChannel2))>255):
                denoisedImageChannel0 = np.divide(denoisedImageChannel0,2)
                denoisedImageChannel1 = np.divide(denoisedImageChannel1,2)
                denoisedImageChannel2 = np.divide(denoisedImageChannel2,2)
                image.append(denoisedImageChannel0.reshape((self.Nx, self.Ny)))
                image.append(denoisedImageChannel1.reshape((self.Nx, self.Ny)))
                image.append(denoisedImageChannel2.reshape((self.Nx, self.Ny)))
            image = image
            
        else:
            image = self.denoisedImage
        #For now since I commented out normalizedTo8Bits
        #image = np.swapaxes(np.array(image), 0, 1)
        #image = np.swapaxes(image,1,2)
        self.image = image


    def normalizeTo8Bits(self):
        image = self.image
        imageOutput = []
        for iChan in range(self.numChannels):
            imageChannel = image[iChan].flatten()
            imageChannel = np.multiply(np.divide(imageChannel,np.max(np.absolute(imageChannel))),255)
            imageOutput.append(imageChannel.reshape((self.Nx, self.Ny)))
        imageOutput = np.swapaxes(np.array(imageOutput), 0, 1)
        imageOutput = np.swapaxes(imageOutput,1,2)
        self.image = imageOutput

    def timeIntegrate(self):
        timeSteps = int(self.finalTime/self.dt)
        timeSteps = 1500
        
        for iTimeStep in range(timeSteps+1):
            print(iTimeStep)
            denoisedImage = []
            noisyImage = self.image
            self.addBoundary()
            self.assignMembership()
            self.createFuzzyMembershipImage()
            self.findFuzzyDerivativeRule()
            self.calculateGradient()
            self.calculateCoefficients()
            self.solveRHS() 
            for iChan in range(self.numChannels):
                denoisedImage.append(noisyImage[:,:,iChan] + self.dt*self.lambd*self.RHS[iChan])
            self.denoisedImage = np.array(denoisedImage)
            self.checkSaturation()
            self.normalizeTo8Bits()
            '''print('Here')
            np.savetxt('../data/outputColorImage/noisyImage0_'+str(iTimeStep)+'.csv',  noisyImage[:,:,0], delimiter=",")
            np.savetxt('../data/outputColorImage/noisyImage1_'+str(iTimeStep)+'.csv',  noisyImage[:,:,1], delimiter=",")
            np.savetxt('../data/outputColorImage/noisyImage2_'+str(iTimeStep)+'.csv',  noisyImage[:,:,2], delimiter=",")
            np.savetxt('../data/outputColorImage/RHS0_'+str(iTimeStep)+'.csv',  self.RHS[0], delimiter=",")
            np.savetxt('../data/outputColorImage/RHS1_'+str(iTimeStep)+'.csv',  self.RHS[1], delimiter=",")
            np.savetxt('../data/outputColorImage/RHS2_'+str(iTimeStep)+'.csv',  self.RHS[2], delimiter=",")
            a = input('').split(" ")[0]'''
            

            #cv2.imwrite('../data/outputColorImage5/denoisedImage'+str(iTimeStep)+'.jpg', gaussian_filter(self.image, sigma=0.2))
            #a = input('').split(" ")[0]
            cv2.imwrite('../data/outputColorImage1/denoisedImage'+str(iTimeStep)+'.jpg', self.image)
            

        self.denoisedImage = noisyImage

    def solve(self):
        self.createPDDOKernelMesh()
        self.findNeighboringPixels()
        self.loadMembershipFunction()
        self.loadGradientMembershipFunctions()
        self.timeIntegrate()
        #a = input('').split(" ")[0]
