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
        self.pathToMembershipFunction = pathToMembershipFunction 
        self.Nx = constants.NX
        self.Ny = constants.NY
        self.horizon = constants.HORIZON
        self.finalTime = constants.FINALTIME
        self.dt = constants.DELTAT
        self.lambd = constants.LAMBDA
        self.GMask = constants.GMASK
        self.g10 = constants.g10
        self.g01 = constants.g01

    def loadMembershipFunction(self):
        self.membershipFunction = np.loadtxt(self.pathToMembershipFunction, delimiter=",")

    def calculateGradient(self):
        self.gradient10 = signal.convolve2d(self.image, self.g10, boundary='symm')
        self.gradient01 = signal.convolve2d(self.image, self.g01, boundary='symm')
    
    def findSimilarityPercent(self, pixelDifferences):
        indexOfMembership = 2 - self.membershipFunction[pixelDifferences,0]
        similarityMembership = []
        for iPixel in range(len(indexOfMembership)):
            idxMembership = indexOfMembership[iPixel]
            if idxMembership == 2.0:
                similarityMembership.append(1.0)
            elif idxMembership == 1.0:
                similarityMembership.append(0.5)
            elif idxMembership == 0.0:
                 similarityMembership.append(0.0)
        similarityPercent = np.divide(np.dot(similarityMembership, self.membershipFunction[pixelDifferences,1] ),np.sum(indexOfMembership))
        return similarityPercent
    
    def findFuzzySimilarityImage(self):
        image = np.pad(self.image,int(self.horizon),mode='symmetric').astype(float)
        similarityPercent = []
        for iCol in range(int(self.horizon),self.Nx+2):
            for iRow in range(int(self.horizon),self.Ny+2):
                pixelDifferences = np.abs(image[iRow,iCol]-image[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]).flatten().astype(int)
                similarityPercent.append(self.findSimilarityPercent(pixelDifferences))
        self.similarityPercent = np.array(similarityPercent).reshape((self.Nx, self.Ny))
        self.similarityImage = np.transpose(np.divide(signal.convolve2d(self.similarityPercent, self.GMask, boundary='symm', mode='same'),len(self.GMask.flatten())))


    def solveRHS(self):
        RHS = []
        gradient = self.gradient10 + self.gradient01
        similarity = np.pad(self.similarityImage, int(self.horizon),'constant')
        for iCol in range(int(self.horizon),self.Nx+2):
            for iRow in range(int(self.horizon),self.Ny+2):
                RHS.append(np.sum(np.multiply(np.multiply(self.GMask,gradient[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1]),similarity[iRow-int(self.horizon):iRow+int(self.horizon)+1,iCol-int(self.horizon):iCol+int(self.horizon)+1])))
        
        self.RHS = np.transpose(np.array(RHS).reshape((self.Nx,self.Ny)))

    def solveDifferentialEquation(self):
        noisyImage = self.image
        self.calculateGradient()
        self.findFuzzySimilarityImage()
        self.solveRHS()
        self.denoisedImage = noisyImage + self.dt*self.lambd*self.RHS


    def timeIntegrate(self):
        timeSteps = int(self.finalTime/self.dt)
        timeSteps = 50
        
        for iTimeStep in range(timeSteps+1):
            print(iTimeStep)
            self.solveDifferentialEquation()           
            self.image = match_histograms(self.image, self.referenceImage)
            np.savetxt('../data/output/'+str(iTimeStep)+'_'+'fuzzySimilarityImage.csv', self.similarityImage)
            np.savetxt('../data/output/'+str(iTimeStep)+'_'+'denoisedImage.csv', self.image)
            #cv2.imwrite('../data/output/'+str(iTimeStep)+'_'+'denoisedImage.jpg', self.image)
            #print(type(img_rescale))
            #a = input('').split(" ")[0]
            #self.image = self.denoisedImage

    def solve(self, referenceImage, image):
        self.referenceImage = referenceImage
        self.image = image
        self.loadMembershipFunction()
        self.timeIntegrate()
        #a = input('').split(" ")[0]
