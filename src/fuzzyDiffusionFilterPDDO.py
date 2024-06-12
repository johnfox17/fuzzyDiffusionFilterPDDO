import constants
import numpy as np
from sklearn.neighbors import KDTree
from PIL import Image

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

    def calculateLocalAndGeneralSmoothness(self):
        localSmoothness = []
        generalAverage = []
        for iChan in range(self.numChannels):
            localSmoothnessChannel = []
            generalAverageChannel = []
            for currentSimilarityMatrix in range((self.Nx-int(2*self.horizon))*(self.Ny-int(2*self.horizon))):
                localSmoothnessChannel.append((np.sum(self.similarityMatrices[iChan][currentSimilarityMatrix].flatten())-1)/(len(self.similarityMatrices[iChan][currentSimilarityMatrix].flatten())-1))
                generalAverageChannel.append(np.average(self.similarityMatrices[iChan][currentSimilarityMatrix]))
            localSmoothness.append(np.transpose(np.array(localSmoothnessChannel).reshape((self.Nx-int(2*self.horizon)),(self.Ny-int(2*self.horizon)))))
            generalAverage.append(np.array(generalAverageChannel).reshape((self.Nx-int(2*self.horizon)),(self.Ny-int(2*self.horizon))))
        self.localSmoothness = np.array(localSmoothness)
        self.generalAverage = np.array(generalAverage)

    def thresholdLocalSmoothness(self):
        for iChan in range(self.numChannels):
            self.localSmoothness[iChan][self.localSmoothness[iChan]>self.threshold] = 0
        
        #print('Here')
        #a = input('').split(" ")[0]

    def solveRHS(self):
        RHS = []
        for iChan in range(self.numChannels):
            gradientChannel = np.pad(self.gradient[iChan].reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))),int(self.horizon),mode='symmetric')
            localSmoothnessChannel = np.pad(self.localSmoothness[iChan] ,int(self.horizon),mode='symmetric')
            RHSChannel = []
            for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
                for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                    RHSChannel.append(np.sum(np.multiply(np.multiply(self.GMask, localSmoothnessChannel[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]),np.multiply(self.GMask,gradientChannel[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]))))
            while np.max(np.absolute(RHSChannel))>255:
                RHSChannel = np.divide(RHSChannel,2)
            RHS.append(np.transpose(np.array(RHSChannel).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon)))))
        self.RHS = np.array(RHS)   

    def checkSaturation(self):       
        image = []
        self.Nx = self.Nx - 2*int(self.horizon)
        self.Ny = self.Ny - 2*int(self.horizon)
        for iChan in range(self.numChannels):
            denoisedImageChannel = self.denoisedImage[iChan].flatten()
            while np.max(np.absolute(denoisedImageChannel))>255:
                denoisedImageChannel = np.divide(denoisedImageChannel,2)
            image.append(denoisedImageChannel.reshape((self.Nx, self.Ny)))
        self.image = image
        
    
    def normalizeTo8Bits(self):
        image = []
        for iChan in range(self.numChannels):
            imageChannel = self.image[iChan].flatten()
            imageChannel = np.multiply(np.divide(imageChannel,np.max(np.absolute(imageChannel))),255)
            image.append(imageChannel.reshape((self.Nx, self.Ny)))
        self.image = np.array(image)

    def timeIntegrate(self):
        timeSteps = int(self.finalTime/self.dt)
        timeSteps = 1500
        
        denoisedImage = []
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
            for iChan in range(self.numChannels):
                denoisedImage.append(noisyImage[:,:,iChan] + self.dt*self.lambd*self.RHS[iChan])
            self.denoisedImage = denoisedImage
            
            self.checkSaturation()
            self.normalizeTo8Bits()
            #Image.fromarray(self.image).save("../data/output/threshold_"+str(self.threshold)+"/denoisedImage"+str(iTimeStep)+".jpeg")
            #Image.fromarray(self.gradient).save("../data/output/threshold_"+str(self.threshold)+"/gradient"+str(iTimeStep)+".jpeg")
            #Image.fromarray(self.localSmoothness).save("../data/output/threshold_"+str(self.threshold)+"/localSmoothness"+str(iTimeStep)+".jpeg")
            #Image.fromarray(self.RHS).save("../data/output/threshold_"+str(self.threshold)+"/RHS"+str(iTimeStep)+".jpeg")
            #a = input('').split(" ")[0]

            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/denoisedImage'+str(iTimeStep)+'0'+'.csv',  self.image[0], delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/gradient'+str(iTimeStep)+'0'+'.csv',  self.gradient[0], delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/localSmoothness'+str(iTimeStep)+'0'+'.csv',  self.localSmoothness[0], delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/RHS'+str(iTimeStep)+'0'+'.csv',  self.RHS[0], delimiter=",")

            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/denoisedImage'+str(iTimeStep)+'1'+'.csv',  self.image[1], delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/gradient'+str(iTimeStep)+'1'+'.csv',  self.gradient[1], delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/localSmoothness'+str(iTimeStep)+'1'+'.csv',  self.localSmoothness[1], delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/RHS'+str(iTimeStep)+'1'+'.csv',  self.RHS[1], delimiter=",")

            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/denoisedImage'+str(iTimeStep)+'2'+'.csv',  self.image[2], delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/gradient'+str(iTimeStep)+'2'+'.csv',  self.gradient[2], delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/localSmoothness'+str(iTimeStep)+'2'+'.csv',  self.localSmoothness[2], delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/RHS'+str(iTimeStep)+'2'+'.csv',  self.RHS[2], delimiter=",")

            print('Here')
            a = input('').split(" ")[0]
        self.denoisedImage = noisyImage

    def solve(self):
        self.createPDDOKernelMesh()
        self.findNeighboringPixels()
        self.loadMembershipFunction()
        self.loadGradientMembershipFunctions()
        self.timeIntegrate()
        #a = input('').split(" ")[0]
