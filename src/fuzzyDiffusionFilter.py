import constants
import numpy as np
from sklearn.neighbors import KDTree

class fuzzyDiffusionFilter:
    def __init__(self, image, pathToMembershipFunction):
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
        self.GxMask = constants.GXMASK
        self.GyMask = constants.GYMASK
        self.gCenter = constants.GCENTER
        self.threshold = constants.THRESHOLD

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
        self.neighboringPixels = tree.query_radius(self.coordinateMesh, r = self.dx*self.horizon)

    
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
                if membershipIndex > 255:
                    membershipIndex = 255
                currentPixelMembership.append(list(self.membershipFunction[membershipIndex])[0])
                currentPixelMembership.append(list(self.membershipFunction[membershipIndex])[1])
                #print(currentPixelMembership)
                #a = input('').split(" ")[0]
                pixelMemberships.append(currentPixelMembership)
        self.pixelMemberships = pixelMemberships

    def createFuzzyMembershipImage(self):
        fuzzyMembershipImage = []
        for iPixel in range(self.Nx*self.Ny):
            fuzzyMembershipImage.append(self.pixelMemberships[iPixel][1])
        self.fuzzyMembershipImage = np.array(fuzzyMembershipImage).reshape((self.Nx,self.Ny))

    def findeFuzzyDerivativeRule(self):
        Dx = []
        Dy = []
        for iCol in range(1,self.Nx-1):
            for iRow in range(1,self.Ny-1):
                Gx = np.multiply(self.GxMask,self.fuzzyMembershipImage[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1])
                Gy = np.multiply(self.GyMask,self.fuzzyMembershipImage[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1])
                Dxi = (Gx[2][2]+Gx[2][1]+Gx[2][0])-(Gx[0][2]+Gx[0][1]+Gx[0][0])
                Dyi = (Gy[2][2]+Gy[1][2]+Gy[0][2])-(Gy[2][0]+Gy[1][0]+Gy[0][0])
                Dx.append(Dxi)
                Dy.append(Dyi)
        self.Dx = np.pad(np.array(Dx).reshape((self.Nx-2, self.Ny-2)),int(self.horizon),mode='symmetric')
        self.Dy = np.pad(np.array(Dy).reshape((self.Nx-2, self.Ny-2)),int(self.horizon),mode='symmetric')

    def calculateGradient(self):
        gX = []
        gY = []
        for iCol in range(1,self.Nx-1):
            for iRow in range(1,self.Ny-1):
                Dx = np.multiply(self.GxMask,self.Dx[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int)
                Dy = np.multiply(self.GyMask,self.Dy[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int)
                Lx = np.multiply(self.GxMask,self.image[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int)
                Lx[Lx>255] = 255
                Ly = np.multiply(self.GyMask,self.image[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int)
                Ly[Ly>255] = 255

                muPremX = self.membershipFunction[int(Lx[2][2])][1]+self.membershipFunction[int(Lx[2][1])][1]+self.membershipFunction[int(Lx[2][0])][1]+self.membershipFunction[int(Lx[0][2])][1]+self.membershipFunction[int(Lx[0][1])][1]+self.membershipFunction[int(Lx[0][0])][1]
                muPremY = self.membershipFunction[Lx[2][2]][1]+self.membershipFunction[Lx[1][2]][1]+self.membershipFunction[Lx[0][2]][1]+self.membershipFunction[Lx[2][0]][1]+self.membershipFunction[Lx[1][0]][1]+self.membershipFunction[Lx[0][0]][1]
                gX.append((self.gCenter[int(Dx[2][2])+6]*self.membershipFunction[Lx[2][2]][1]+\
                        self.gCenter[int(Dx[2][1])+6]*self.membershipFunction[Lx[2][1]][1]+\
                        self.gCenter[int(Dx[2][0])+6]*self.membershipFunction[Lx[2][0]][1]+\
                        self.gCenter[int(Dx[0][2])+6]*self.membershipFunction[Lx[0][2]][1]+\
                        self.gCenter[int(Dx[0][1])+6]*self.membershipFunction[Lx[0][1]][1]+\
                        self.gCenter[int(Dx[0][0])+6]*self.membershipFunction[Lx[0][0]][1])/muPremX)

                gY.append((self.gCenter[int(Dy[2][2])+6]*self.membershipFunction[Lx[2][2]][1]+\
                        self.gCenter[int(Dy[1][2])+6]*self.membershipFunction[Lx[1][2]][1]+\
                        self.gCenter[int(Dy[0][2])+6]*self.membershipFunction[Lx[0][2]][1]+\
                        self.gCenter[int(Dy[2][0])+6]*self.membershipFunction[Lx[2][0]][1]+\
                        self.gCenter[int(Dy[1][0])+6]*self.membershipFunction[Lx[1][0]][1]+\
                        self.gCenter[int(Dy[0][0])+6]*self.membershipFunction[Lx[0][0]][1])/muPremY)
        self.gX = np.array(gX)
        self.gY = np.array(gY)
        self.g = self.gX+self.gY
    

    def createSimilarityMatrices(self):
        similarityMatrices = []
        for iCol in range(1,self.Nx-1):
            for iRow in range(1,self.Ny-1):
                similarityMatrices.append(np.exp(-np.power(np.absolute(np.multiply(self.GxMask,self.image[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1])-self.image[iRow,iCol]),self.q)/self.Dn))
        similarityMatrices = np.array(similarityMatrices)
        similarityMatrices[similarityMatrices<1e-9] = 0
        self.similarityMatrices = similarityMatrices

    def solveRHS(self):
        g = np.pad(self.g.reshape((self.Nx-2,self.Ny-2)) ,int(self.horizon),mode='symmetric')
        localSmoothness = np.pad(self.localSmoothness ,int(self.horizon),mode='symmetric')
        #iPixel = 0
        RHS = []
        for iCol in range(1,self.Nx-1):
            for iRow in range(1,self.Ny-1):
                #RHS.append(np.sum(np.multiply(self.similarityMatrices[iPixel,:,:],np.multiply(self.GxMask,g[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]))))           
                RHS.append(np.sum(np.multiply(np.multiply(self.GxMask, localSmoothness[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]),np.multiply(self.GxMask,g[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]))))
                #iPixel = iPixel + 1
        self.RHS = np.transpose(np.array(RHS).reshape((self.Nx-2,self.Ny-2)))

    

    def calculateLocalAndGeneralSmoothness(self):
        localSmoothness = []
        generalAverage = []
        for currentSimilarityMatrix in range((self.Nx-2)*(self.Ny-2)):
            localSmoothness.append((np.sum(self.similarityMatrices[currentSimilarityMatrix])-1)/(len(self.similarityMatrices[currentSimilarityMatrix])-1))
            generalAverage.append(np.average(self.similarityMatrices[currentSimilarityMatrix]))
        self.localSmoothness = np.transpose(np.array(localSmoothness).reshape((self.Nx-2),(self.Ny-2)))
        self.generalAverage = np.array(generalAverage).reshape((self.Nx-2),(self.Ny-2))

    def thresholdLocalSmoothness(self):
        localSmoothness = np.array(self.localSmoothness)
        localSmoothness[localSmoothness<self.threshold] = 1
        localSmoothness[localSmoothness != 1] = 0
        self.localSmoothness = localSmoothness

    def checkSaturation(self):       
        denoisedImage = self.denoisedImage.flatten()
        while np.max(np.absolute(denoisedImage))>256:
            denoisedImage = np.divide(denoisedImage,2)
        
        self.Nx = self.Nx - 2*int(self.horizon)
        self.Ny = self.Ny - 2*int(self.horizon)
        self.image = denoisedImage.reshape((self.Nx, self.Ny))

    def timeIntegrate(self):
        timeSteps = int(self.finalTime/self.dt)
        timeSteps = 300
        for iTimeStep in range(timeSteps+1):
            print(iTimeStep)
            noisyImage = self.image
            self.addBoundary()
            self.assignMembership()
            self.createFuzzyMembershipImage()
            self.findeFuzzyDerivativeRule()
            self.calculateGradient() 
            self.createSimilarityMatrices()
            self.calculateLocalAndGeneralSmoothness()
            self.thresholdLocalSmoothness()
            self.solveRHS() 
            self.denoisedImage = noisyImage + self.dt*self.lambd*self.RHS
            self.checkSaturation()

            #if iTimeStep%10 == 0:
                #np.savetxt('..\\data\\denoisedImage'+str(iTimeStep)+'.csv',  self.image, delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/denoisedImage'+str(iTimeStep)+'.csv',  self.image, delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/g'+str(iTimeStep)+'.csv',  self.g, delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/localSmoothness'+str(iTimeStep)+'.csv',  self.localSmoothness, delimiter=",")
            np.savetxt('../data/output/threshold_'+str(self.threshold)+'/RHS'+str(iTimeStep)+'.csv',  self.RHS, delimiter=",")
        self.denoisedImage = noisyImage

    def solve(self):
        self.createPDDOKernelMesh()
        self.findNeighboringPixels()
        self.loadMembershipFunction()
        self.loadGradientMembershipFunctions()
        self.timeIntegrate()
        #a = input('').split(" ")[0]
