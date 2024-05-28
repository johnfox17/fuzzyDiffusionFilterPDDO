import constants
import numpy as np
from sklearn.neighbors import KDTree

class fuzzyDiffusionFilterPDDO:
    def __init__(self, image, pathToMembershipFunction, threshold):
        self.image = image
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
        self.image = np.pad(self.image,int(self.horizon),mode='symmetric')
        self.Nx = self.Nx + 2*int(self.horizon)
        self.Ny = self.Ny + 2*int(self.horizon)

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
        for iCol in range(self.Nx):
            for iRow in range(self.Ny):
                currentPixelMembership = []
                currentPixelMembership.append(self.image[iCol,iRow])
                membershipIndex = int(self.image[iCol,iRow])
                currentPixelMembership.append(list(self.membershipFunction[membershipIndex])[0])
                currentPixelMembership.append(list(self.membershipFunction[membershipIndex])[1])
                pixelMemberships.append(currentPixelMembership)
        self.pixelMemberships = pixelMemberships

    def createFuzzyMembershipImage(self):
        fuzzyMembershipImage = []
        for iPixel in range(self.Nx*self.Ny):
            fuzzyMembershipImage.append(self.pixelMemberships[iPixel][1])
        self.fuzzyMembershipImage = np.array(fuzzyMembershipImage).reshape((self.Nx,self.Ny))

    def findFuzzyDerivativeRule(self):
        D = []
        for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
            for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                D.append(np.sum(np.multiply(self.g,self.fuzzyMembershipImage[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).flatten()).astype(int))
        D = np.array(D)
        while np.max(np.absolute(D))>255:
            D = np.divide(D,2)
        self.D = np.pad(D.reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))),int(self.horizon),mode='symmetric') 
        #np.savetxt('../data/output/DerivativeRule2.csv',  self.D, delimiter=",")
        #print('Here')
        #a = input('').split(" ")[0]

    def calculateGradient(self):
        gradient = []
        #print(self.g)
        #print(self.GMask)
        for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
            for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                D = np.multiply(self.GMask,self.D[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int).flatten()
                L = np.multiply(self.GMask,self.image[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int).flatten()
                muPrem = np.sum(self.membershipFunction[L,1])
                gCents = []
                for iD in D:
                    #print(iD)
                    gCents.append(self.gCenter[np.abs(self.gCenter-iD).argmin()])
                    #a = input('').split(" ")[0]
                gCents = np.array(gCents).reshape((self.kerneldim,self.kerneldim))
                gradient.append(np.sum(np.multiply(self.GMask,(np.multiply(gCents, self.membershipFunction[L,1].reshape((self.kerneldim,self.kerneldim))))/muPrem).flatten()))
        while np.max(np.absolute(gradient))>255:
            gradient = np.divide(gradient,2)
        self.gradient = np.array(gradient)

    def createSimilarityMatrices(self):
        similarityMatrices = []
        for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
            for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                similarityMatrices.append(np.exp(-np.power(np.absolute(np.multiply(self.GMask,self.image[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1])-self.image[iRow,iCol]),self.q)/self.Dn))
        similarityMatrices = np.array(similarityMatrices)
        similarityMatrices[similarityMatrices<1e-9] = 0
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
