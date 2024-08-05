import numpy as np

L1 = 1
L2 = 1
NX = 256
NY = 256
NUMCHANNELS = 3
#HORIZON = 3.015
HORIZON = 2.015
q = 2
Dn = 2**8-1
FINALTIME = 100
#DELTAT = 0.004
DELTAT = 0.1
LAMBDA = 0.5
#THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
THRESHOLDS = [0.2]
KERNELDIM = 5
DINDEXOFFSET = 6
GMASK = np.ones((KERNELDIM, KERNELDIM))
GXMASK = np.array([[1,1,1],[0,0,0],[1,1,1]])
GYMASK = np.array([[1,0,1],[1,0,1],[1,0,1]])
RHSMASK = np.array([[0,1,0],[1,0,1],[0,1,0]])

D_NEG6 = np.loadtxt('../data/simData/D_-6.csv', delimiter=",") 
D_NEG5 = np.loadtxt('../data/simData/D_-5.csv', delimiter=",")
D_NEG4 = np.loadtxt('../data/simData/D_-4.csv', delimiter=",")
D_NEG3 = np.loadtxt('../data/simData/D_-3.csv', delimiter=",")
D_NEG2 = np.loadtxt('../data/simData/D_-2.csv', delimiter=",")
D_NEG1 = np.loadtxt('../data/simData/D_-1.csv', delimiter=",")
D_0 = np.loadtxt('../data/simData/D_0.csv', delimiter=",")
D_1 = np.loadtxt('../data/simData/D_1.csv', delimiter=",")
D_2 = np.loadtxt('../data/simData/D_2.csv', delimiter=",")
D_3 = np.loadtxt('../data/simData/D_3.csv', delimiter=",")
D_4 = np.loadtxt('../data/simData/D_4.csv', delimiter=",")
D_5 = np.loadtxt('../data/simData/D_5.csv', delimiter=",")
D_6 = np.loadtxt('../data/simData/D_6.csv', delimiter=",")

gGradient = np.loadtxt('../data/simData/gGradient.csv', delimiter=",")
g10 = np.loadtxt('../data/simData/g10.csv', delimiter=",")
g01 = np.loadtxt('../data/simData/g01.csv', delimiter=",")
contrastEnhancementLUT = np.loadtxt('../data/simData/contrastEnhancementLookUpTable.csv', delimiter=",")
#GCENTER = [-255,-212.5,-170,-127.5,-85,-42.5,0,42.5,85,127.5,170,212.5,255]
GCENTER = np.array([-255, -244.80, -234.60, -224.40, -214.20, -204, -193.80, -183.60, -173.40, -163.20, -153, -142.80, -132.60, -122.40, -112.20, -102, -91.80, -81.60, -71.40, -61.20, -51, -40.80, -30.60, -20.40, -10.20, 0, 10.20, 20.40, 30.60, 40.80, 51, 61.20, 71.40, 81.60, 91.80, 102, 112.20, 122.40, 132.60, 142.80, 153, 163.20, 173.40, 183.60, 193.80, 204.00, 214.20, 224.40, 234.60, 244.80, 255.00])
#.flatten().astype(int)
