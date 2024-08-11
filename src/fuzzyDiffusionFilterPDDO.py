import constants
import numpy as np
from sklearn.neighbors import KDTree
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
import math
from skimage import exposure

class fuzzyDiffusionFilterPDDO:
    def __init__(self,pathToMembershipFunction):
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
        self.image = np.pad(self.image,int(self.horizon),mode='symmetric')
        #self.image = np.pad(self.image,int(self.horizon),'constant')
    
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
                currentPixelMembership.append(self.image[iRow,iCol])
                membershipIndex = int(math.floor(self.image[iRow,iCol]))
                currentPixelMembership.append(list(self.membershipFunction[membershipIndex])[0])
                currentPixelMembership.append(list(self.membershipFunction[membershipIndex])[1])
                pixelMemberships.append(currentPixelMembership)
        self.pixelMemberships = np.array(pixelMemberships)

    def createFuzzyMembershipImage(self):
        self.fuzzyMembershipImage = np.transpose(np.array(self.pixelMemberships[:,1].reshape((self.Nx,self.Ny))))

    def findFuzzyDerivativeRule(self):
        D10 = []
        D01 = []
        for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
            for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                D10.append(np.sum(np.multiply(self.g10,self.fuzzyMembershipImage[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).flatten()).astype(int))
                D01.append(np.sum(np.multiply(self.g01,self.fuzzyMembershipImage[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).flatten()).astype(int))
        self.D10 = np.transpose(np.pad(np.array(D10).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))),int(self.horizon),mode='symmetric'))
        self.D01 = np.transpose(np.pad(np.array(D01).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))),int(self.horizon),mode='symmetric'))

        #np.savetxt('../data/output/D10.csv',  self.D10, delimiter=",")
        #np.savetxt('../data/output/D01.csv',  self.D10, delimiter=",")
        #print('Here')
        #a = input('').split(" ")[0]
        
    def calculateGradient(self):
        gradient10 = []
        gradient01 = []
        for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
            for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                D10 = np.multiply(self.GMask,self.D10[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int).flatten()
                D01 = np.multiply(self.GMask,self.D01[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int).flatten()
                L = np.multiply(self.GMask,self.image[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int).flatten()
                muPrem = np.sum(self.membershipFunction[L,1])#This might have to be changed to include all channels of image
                gCents10 = []
                for iD in D10:
                    gCents10.append(self.gCenter[np.abs(self.gCenter-iD).argmin()])
                gradient10.append(np.sum(np.multiply(np.array(gCents10), self.membershipFunction[L,1]))/muPrem)
                gCents01 = []
                for iD in D01:
                    gCents01.append(self.gCenter[np.abs(self.gCenter-iD).argmin()])
                gradient01.append(np.sum(np.multiply(np.array(gCents01), self.membershipFunction[L,1]))/muPrem)
        self.gradient10 = np.transpose(np.pad(np.array(gradient10).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))),int(self.horizon),mode='symmetric'))
        self.gradient01 = np.transpose(np.pad(np.array(gradient01).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))),int(self.horizon),mode='symmetric'))
        #self.gradient10 = np.divide(gradient10, np.linalg.norm(gradient10))
        #self.gradient01 = np.divide(gradient01, np.linalg.norm(gradient01))
    def calculateCoefficients(self):
        coefficients = [] 
        #K=0.8*np.mean(self.gradient10 + self.gradient01)
        K=0.8
        for iCol in range(self.Nx-int(2*self.horizon)):
            for iRow in range(self.Ny-int(2*self.horizon)):
                iGradientMagnitude = np.sqrt(self.gradient10[iRow,iCol]**2 + self.gradient01[iRow,iCol]**2)
                #iGradientMagnitude = np.sqrt(self.gradient10[iRow,iCol]**2)
                coefficients.append(np.exp(-np.power(np.abs(np.divide(iGradientMagnitude,K)),2)))
                #coefficients.append(1/(1+np.divide(iGradientMagnitude,K)))
        self.coefficients = np.array(np.pad(np.transpose(np.array(coefficients).reshape((self.Nx-int(2*self.horizon), self.Ny-int(2*self.horizon)))),int(self.horizon),'constant'))
        #self.coefficients = np.array(np.transpose(np.array(coefficients).reshape((self.Nx-int(2*self.horizon), self.Ny-int(2*self.horizon)))))
        #self.coefficients = np.divide(coefficients, np.max(coefficients))

    def calculateGradientOfCoefficients(self):
        gradientCoefficients10 = []
        gradientCoefficients01 = []
        for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
            for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                D10 = np.multiply(self.GMask,self.D10[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int).flatten()
                D01 = np.multiply(self.GMask,self.D01[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int).flatten()
                L = np.multiply(self.GMask,self.coefficients[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).astype(int).flatten()
                muPrem = np.sum(self.membershipFunction[L,1])#This might have to be changed to include all channels of image
                gCents10 = []
                for iD in D10:
                    gCents10.append(self.gCenter[np.abs(self.gCenter-iD).argmin()])
                gradientCoefficients10.append(np.sum(np.multiply(np.array(gCents10), self.membershipFunction[L,1]))/muPrem)
                gCents01 = []
                for iD in D01:
                    gCents01.append(self.gCenter[np.abs(self.gCenter-iD).argmin()])
                gradientCoefficients01.append(np.sum(np.multiply(np.array(gCents01), self.membershipFunction[L,1]))/muPrem)

        self.gradientCoefficients10 = np.transpose(np.array(gradientCoefficients10).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))))
        self.gradientCoefficients01 = np.transpose(np.array(gradientCoefficients01).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))))
        #self.gradientCoefficients10 = np.divide(gradientCoefficients10, np.linalg.norm(gradientCoefficients10))
        #self.gradientCoefficients01 = np.divide(gradientCoefficients01, np.linalg.norm(gradientCoefficients01))
        #a = input('').split(" ")[0]

    def solveRHS(self):
        #self.RHS = np.multiply(self.gradient10,self.gradientCoefficients10) + np.multiply(self.gradient01,self.gradientCoefficients01)
        #self.RHS = np.multiply(self.gradient10,self.coefficients) + np.multiply(self.gradient01,self.coefficients)
        RHS = []
        gradient = self.gradient10 + self.gradient01
        for iCol in range(int(self.horizon),self.Nx-int(self.horizon)):
            for iRow in range(int(self.horizon),self.Ny-int(self.horizon)):
                RHS.append(np.sum(np.multiply(np.multiply(self.GMask,gradient[iRow-int(self.horizon-2):iRow+int(self.horizon-2)+1,iCol-int(self.horizon-2):iCol+int(self.horizon-2)+1]),self.coefficients[iRow-int(self.horizon-2):iRow+int(self.horizon-2)+1,iCol-int(self.horizon-2):iCol+int(self.horizon-2)+1])))
        
        self.RHS = np.transpose(np.array(RHS).reshape((self.Nx-int(2*self.horizon),self.Ny-int(2*self.horizon))))


    def checkSaturation(self):       
        self.Nx = self.Nx - 2*int(self.horizon)
        self.Ny = self.Ny - 2*int(self.horizon)
        image = self.denoisedImage.flatten()
        self.saturated = False
        while (np.max(np.absolute(image))>255 ):
            image = np.divide(image,2)
            self.saturated = True
        self.image = image

    def normalizeTo8Bits(self):
        #self.image = np.multiply(np.divide(self.image,np.max(np.absolute(self.image))),255).reshape((self.Nx, self.Ny))
        self.image = self.image.reshape((self.Nx, self.Ny))

    def enhanceContrast(self):
        image = []
        for iCol in range(self.Nx):
            for iRow in range(self.Ny):
                image.append(self.contrastEnhancementLUT[int(self.image[iRow,iCol])])
        self.image = np.transpose(np.array(image).reshape((self.Nx, self.Ny)))

    def solveDifferentialEquation(self):
        noisyImage = self.image
        #print(np.shape(self.image))
        self.addBoundary()
        #print('Add Boundary '+ str(np.shape(self.image)))
        self.assignMembership()
        #print('Assign Membership' + str(np.shape(self.image)))
        self.createFuzzyMembershipImage()
        #print('Create Fuzzy Membership '+ str(np.shape(self.image)))
        self.findFuzzyDerivativeRule()
        #print('findFuzzyDerivativeRule '+ str(np.shape(self.image)))
        self.calculateGradient()
        #print('calculateGradient'+str(np.shape(self.image)))
        self.calculateCoefficients()
        #print('calculateCoefficients'+str(np.shape(self.image)))
        self.calculateGradientOfCoefficients()
        #print('calculateGradientOfCoefficients '+str(np.shape(self.image)))
        self.solveRHS()
        #print('solveRHS'+str(np.shape(self.image)))
        self.denoisedImage = noisyImage + self.dt*self.lambd*self.RHS

    def timeIntegrate(self):
        timeSteps = int(self.finalTime/self.dt)
        timeSteps = 1500
        
        for iTimeStep in range(timeSteps+1):
            print(iTimeStep)
            self.solveDifferentialEquation()           
            self.checkSaturation()
            self.normalizeTo8Bits()
            #np.savetxt('../data/outputColorImage7/RHS.csv',  self.RHS, delimiter=",")
            #print('Here')
            #a = input('').split(" ")[0]
            #if self.saturated: 
            #    self.enhanceContrast()
            '''print('Here')
            np.savetxt('../data/outputColorImage/RHS2_'+str(iTimeStep)+'.csv',  self.RHS[2], delimiter=",")
            a = input('').split(" ")[0]'''
            image = np.divide(self.image,2**8) 
            # Contrast stretching
            p0, p1 = np.percentile(image, (10, 99.9))
            self.image = np.multiply(255,exposure.rescale_intensity(image, in_range=(p0, p1)))
            #self.image = exposure.rescale_intensity(self.image, in_range=(p0, p1)).astype(int)
            np.savetxt('../data/output_0.8Mean/'+str(iTimeStep)+'_'+'denoisedImage.csv', self.image)
            #cv2.imwrite('../data/output/'+str(iTimeStep)+'_'+'denoisedImage.jpg', self.image)
            #print(type(img_rescale))
            #a = input('').split(" ")[0]
            #self.image = self.denoisedImage

    def solve(self, image):
        self.image = image
        self.createPDDOKernelMesh()
        self.findNeighboringPixels()
        self.loadMembershipFunction()
        self.loadGradientMembershipFunctions()
        
        self.timeIntegrate()
        #a = input('').split(" ")[0]
