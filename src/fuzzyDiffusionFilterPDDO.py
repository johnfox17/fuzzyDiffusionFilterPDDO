import constants
import numpy as np
from sklearn.neighbors import KDTree
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter
import math
from skimage import exposure
from scipy import signal
from skimage.exposure import match_histograms

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
    

    def calculateGradient(self):
        self.gradient10 = signal.convolve2d(self.image, self.g10, boundary='symm')
        self.gradient01 = signal.convolve2d(self.image, self.g01, boundary='symm')
    
    def calculateCoefficients(self):
        coefficients = [] 
        #K=0.5*np.mean(np.abs(self.gradient10) + np.abs(self.gradient01))
        K=0.8
        for iCol in range(self.Nx):
            for iRow in range(self.Ny):
                iGradientMagnitude = np.sqrt(self.gradient10[iRow,iCol]**2 + self.gradient01[iRow,iCol]**2)
                #iGradientMagnitude = np.sqrt(self.gradient10[iRow,iCol]**2)
                #coefficients.append(np.exp(-np.power(np.abs(np.divide(iGradientMagnitude,K)),2)))
                coefficients.append(1/(1+np.divide(iGradientMagnitude,K)))
        self.coefficients = np.pad(np.array(np.transpose(np.array(coefficients).reshape((self.Nx, self.Ny)))),int(self.horizon),mode='constant')
        #np.savetxt('../data/outputPDDODerivative3/coefficients.csv',  self.coefficients, delimiter=",")
        #print('Here')
        #a = input('').split(" ")[0]

    def calculateGradientOfCoefficients(self):
        self.gradientCoefficients10 = np.pad(signal.convolve2d(self.coefficients, self.g10, boundary='symm', mode='same'),int(self.horizon),'constant')
        self.gradientCoefficients01 = np.pad(signal.convolve2d(self.coefficients, self.g01, boundary='symm', mode='same'),int(self.horizon),'constant')
        #self.gradientCoefficients10 = np.pad(self.coefficients,int(self.horizon),'constant')
        #self.gradientCoefficients01 = np.pad(self.coefficients,int(self.horizon),'constant')

    def solveRHS(self):
        #self.RHS = np.multiply(self.gradient10,self.gradientCoefficients10) + np.multiply(self.gradient01,self.gradientCoefficients01)
        #self.RHS = np.multiply(self.gradient10,self.coefficients) + np.multiply(self.gradient01,self.coefficients)
        RHS = []
        gradient = self.gradient10 + self.gradient01
        for iCol in range(int(self.horizon),self.Nx+2):
            for iRow in range(int(self.horizon),self.Ny+2):
                RHS.append(np.sum(np.multiply(np.multiply(self.GMask,self.gradient10[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]),self.coefficients[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]))+ np.sum(np.multiply(np.multiply(self.GMask,self.gradient01[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]),self.coefficients[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1])))
                #RHS.append(np.sum(np.multiply(np.multiply(self.rhsMask,self.gradient10[iRow-int(self.horizon-2):iRow+int(self.horizon-2)+1,iCol-int(self.horizon-2):iCol+int(self.horizon-2)+1]),self.gradientCoefficients10[iRow-int(self.horizon-2):iRow+int(self.horizon-2)+1,iCol-int(self.horizon-2):iCol+int(self.horizon-2)+1]))+ np.sum(np.multiply(np.multiply(self.rhsMask,self.gradient01[iRow-int(self.horizon-2):iRow+int(self.horizon-2)+1,iCol-int(self.horizon-2):iCol+int(self.horizon-2)+1]),self.gradientCoefficients01[iRow-int(self.horizon-2):iRow+int(self.horizon-2)+1,iCol-int(self.horizon-2):iCol+int(self.horizon-2)+1])))
        
        self.RHS = np.transpose(np.array(RHS).reshape((self.Nx,self.Ny)))


    def checkSaturation(self):       
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
        self.calculateGradient()
        self.calculateCoefficients()
        self.calculateGradientOfCoefficients()        
        self.solveRHS()
        #print('solveRHS'+str(np.shape(self.image)))
        self.denoisedImage = noisyImage + self.dt*self.lambd*self.RHS

    def timeIntegrate(self):
        timeSteps = int(self.finalTime/self.dt)
        timeSteps = 30000
        
        for iTimeStep in range(timeSteps+1):
            print(iTimeStep)
            self.solveDifferentialEquation()           
            self.checkSaturation()
            self.normalizeTo8Bits()
            #if iTimeStep % 100 == 0:
            #p0, p1 = np.percentile(self.image, (10, 99.9))
            #self.image = np.multiply(255,exposure.rescale_intensity(self.image, in_range=(p0, p1)))
            #self.image = match_histograms(self.image, self.referenceImage)
            #np.savetxt('../data/outputColorImage7/RHS.csv',  self.RHS, delimiter=",")
            #print('Here')
            #a = input('').split(" ")[0]
            #if self.saturated: 
            #    self.enhanceContrast()
            '''print('Here')
            np.savetxt('../data/outputColorImage/RHS2_'+str(iTimeStep)+'.csv',  self.RHS[2], delimiter=",")
            a = input('').split(" ")[0]'''
            #image = np.divide(self.image,2**8) 
            # Contrast stretching
            #p0, p1 = np.percentile(image, (10, 99.9))
            #image = np.multiply(255,exposure.rescale_intensity(image, in_range=(p0, p1)))
            #self.image = exposure.rescale_intensity(self.image, in_range=(p0, p1)).astype(int)
            if iTimeStep % 100 == 0:
                image = match_histograms(self.image, self.referenceImage)
                np.savetxt('../data/outputPDDODerivative3/'+str(iTimeStep)+'_'+'denoisedImage.csv', image)
            #cv2.imwrite('../data/output/'+str(iTimeStep)+'_'+'denoisedImage.jpg', self.image)
            #print(type(img_rescale))
            #a = input('').split(" ")[0]
            #self.image = self.denoisedImage

    def solve(self, referenceImage, image):
        self.referenceImage = referenceImage
        self.image = image
        self.createPDDOKernelMesh()
        self.findNeighboringPixels()
        
        self.timeIntegrate()
        #a = input('').split(" ")[0]
