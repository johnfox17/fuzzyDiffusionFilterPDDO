import numpy as np

L1 = 1
L2 = 1
NX = 512
NY = 512
#HORIZON = 3.015
HORIZON = 1.42
q = 2
Dn = 2**8-1
FINALTIME = 100
DELTAT = 0.01
LAMBDA = 0.25
THRESHOLD = 0.02
GXMASK = np.array([[1,1,1],[0,0,0],[1,1,1]])
GYMASK = np.array([[1,0,1],[1,0,1],[1,0,1]])

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

GCENTER = [-255,-212.5,-170,-127.5,-85,-42.5,0,42.5,85,127.5,170,212.5,255]
#.flatten().astype(int)
