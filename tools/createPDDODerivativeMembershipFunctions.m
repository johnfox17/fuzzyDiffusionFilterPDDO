clear;
close all;

grayScaleDerivativeMesh = []; 
smallestValue = -255;
largestValue = 255;
smallestValueOfPDDOFuzzyDerivativeRule = 0;
largestValueOfPDDOFuzzyDerivativeRule = 50;
numOfDivisions = 50;

dx = (largestValue - smallestValue)/numOfDivisions;
for i=0:50
    grayScaleDerivativeMesh=[grayScaleDerivativeMesh, smallestValue+dx*i];
end
