import constants
import numpy as np
from sklearn.neighbors import KDTree

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
        self.g = constants.G
        self.dindexoffset = constants.DINDEXOFFSET
        self.kerneldim = constants.KERNELDIM

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
        self.pixelMemberships = pixelMemberships

    def createFuzzyMembershipImage(self):
        fuzzyMembershipImage = []
        for iChan in range(self.numChannels):
            fuzzyMembershipChannel = []
            for iPixel in range(self.Nx*self.Ny):
                fuzzyMembershipChannel.append(self.pixelMemberships[iChan][iPixel][1])
            fuzzyMembershipImage.append(np.array(fuzzyMembershipChannel).reshape((self.Nx,self.Ny)))
        self.fuzzyMembershipImage = np.array(fuzzyMembershipImage)

    def findFuzzyDerivativeRule(self):
        D = []
        for iChan in range(self.numChannels):
            DChannel = []
            for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
                for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                    DChannel.append(np.sum(np.multiply(self.g,self.fuzzyMembershipImage[iChan, iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).flatten()).astype(int))
            D.append(np.pad(np.array(DChannel).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))),int(self.horizon),mode='symmetric'))
        for iChan in range(self.numChannels):
            while np.max(np.absolute(D[iChan]))>255:
                D[iChan] = np.divide(D[iChan],2)
        self.D = D

    def calculateGradient(self):
        gradient = []
        for iChan in range(self.numChannels):
            gradientChannel = []
            for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
                for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                    D = np.multiply(self.GMask,self.D[iChan][iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int).flatten()
                    L = np.multiply(self.GMask,self.image[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1, iChan]).astype(int).flatten()
                    muPrem = np.sum(self.membershipFunction[L,1])
                    gCents = []
                    for iD in D:
                        gCents.append(self.gCenter[np.abs(self.gCenter-iD).argmin()])
                    gCents = np.array(gCents).reshape((self.kerneldim,self.kerneldim))
                    gradientChannel.append(np.sum(np.multiply(self.GMask,(np.multiply(gCents, self.membershipFunction[L,1].reshape((self.kerneldim,self.kerneldim))))/muPrem).flatten()))
            gradient.append(gradientChannel)
        
        for iChan in range(self.numChannels):
            while np.max(np.absolute(gradient[iChan]))>255:
                gradient[iChan] = np.divide(gradient[iChan],2)
        self.gradient = np.array(gradient)

    def createSimilarityMatrices(self):
        similarityMatrices = []
        for iChan in range(self.numChannels):
            similarityMatricesChannel = []
            for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
                for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                    similarityMatricesChannel.append(np.exp(-np.power(np.absolute(np.multiply(self.GMask,self.image[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1, iChan])-self.image[iRow,iCol,iChan]),self.q)/self.Dn))
            similarityMatricesChannel = np.array(similarityMatricesChannel)
            similarityMatricesChannel[similarityMatricesChannel<1e-9] = 0
            similarityMatrices.append(similarityMatricesChannel) 
        self.similarityMatrices = np.array(similarityMatrices)

    def solveRHS(self):
        gradient = np.pad(self.gradient.reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))),int(self.horizon),mode='symmetric')
        localSmoothness = np.pad(self.localSmoothness ,int(self.horizon),mode='symmetric')
        RHS = []
        for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
            for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                RHS.append(np.sum(np.multiply(np.multiply(self.GMask, localSmoothness[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]),np.multiply(self.GMask,gradient[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]))))
        while np.max(np.absolute(RHS))>255:
            RHS = np.divide(RHS,2)
        print(np.shape(self.similarityMatrices))
        print('Here')
        a = input('').split(" ")[0]
        self.RHS = np.transpose(np.array(RHS).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))))
 

    def calculateLocalAndGeneralSmoothness(self):
        localSmoothness = []
        generalAverage = []
        for currentSimilarityMatrix in range((self.Nx-int(2*self.horizon))*(self.Ny-int(2*self.horizon))):
            localSmoothness.append((np.sum(self.similarityMatrices[currentSimilarityMatrix].flatten())-1)/(len(self.similarityMatrices[currentSimilarityMatrix].flatten())-1))
            generalAverage.append(np.average(self.similarityMatrices[currentSimilarityMatrix]))
        self.localSmoothness = np.transpose(np.array(localSmoothness).reshape((self.Nx-int(2*self.horizon)),(self.Ny-int(2*self.horizon))))
        self.generalAverage = np.array(generalAverage).reshape((self.Nx-int(2*self.horizon)),(self.Ny-int(2*self.horizon)))

    def thresholdLocalSmoothness(self):
        localSmoothness = np.array(self.localSmoothness)
        localSmoothness[localSmoothness>self.threshold] = 0
        #localSmoothness[localSmoothness != 1] = 0
        self.localSmoothness = localSmoothness

    def checkSaturation(self):       
        denoisedImage = self.denoisedImage.flatten()
        while np.max(np.absolute(denoisedImage))>255:
            denoisedImage = np.divide(denoisedImage,2)
        
        self.Nx = self.Nx - 2*int(self.horizon)
        self.Ny = self.Ny - 2*int(self.horizon)
        self.image = denoisedImage.reshape((self.Nx, self.Ny))

    
    def normalizeTo8Bits(self):
        image = self.image.flatten()
        image = np.multiply(np.divide(image,np.max(np.absolute(image))),255)
        self.image = image.reshape((self.Nx, self.Ny))

    def timeIntegrate(self):
        timeSteps = int(self.finalTime/self.dt)
        timeSteps = 1500
        
        for iTimeStep in range(timeSteps+1):
            print(iTimeStep)
            noisyImage = self.image
            self.addBoundary()
            self.assignMembership()
            self.createFuzzyMembershipImage()
            self.findFuzzyDerivativeRule()
            self.calculateGradient()
            self.createSimilarityMatrices()
            np.savetxt('../data/output/gradient0.csv',  self.gradient[0], delimiter=",")
            np.savetxt('../data/output/gradient1.csv',  self.gradient[1], delimiter=",")
            np.savetxt('../data/output/gradient2.csv',  self.gradient[2], delimiter=",")
            print('Here')
            a = input('').split(" ")[0]
            self.calculateLocalAndGeneralSmoothness()
            self.thresholdLocalSmoothness()
            self.solveRHS() 
            self.denoisedImage = noisyImage + self.dt*self.lambd*self.RHS
            self.checkSaturation()
            self.normalizeTo8Bits()

            np.savetxt('../data/output5/threshold_'+str(self.threshold)+'/denoisedImage'+str(iTimeStep)+'.csv',  self.image, delimiter=",")
            np.savetxt('../data/output5/threshold_'+str(self.threshold)+'/gradient'+str(iTimeStep)+'.csv',  self.gradient, delimiter=",")
            np.savetxt('../data/output5/threshold_'+str(self.threshold)+'/localSmoothness'+str(iTimeStep)+'.csv',  self.localSmoothness, delimiter=",")
            np.savetxt('../data/output5/threshold_'+str(self.threshold)+'/RHS'+str(iTimeStep)+'.csv',  self.RHS, delimiter=",")
        self.denoisedImage = noisyImage

    def solve(self):
        self.createPDDOKernelMesh()
        self.findNeighboringPixels()
        self.loadMembershipFunction()
        self.loadGradientMembershipFunctions()
        self.timeIntegrate()
        #a = input('').split(" ")[0]
