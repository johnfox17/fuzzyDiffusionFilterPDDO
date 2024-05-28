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

figure; 
for i=1:length(grayScaleDerivativeMesh)-2
    a = grayScaleDerivativeMesh(i);
    b = grayScaleDerivativeMesh(i+1);
    c = grayScaleDerivativeMesh(i+2);
    L = a:1:b;
    D = (b-L)/(b-a);
    plot(L,D,'k');
    hold on;
    
    L_left = a:1:b;
    L_right = b:1:c;
    D_left = (L_left-a)/(b-a);
    D_right = (c- L_right)/(c-b);
    D = [D_left,D_right];
    L = [L_left,L_right];
    plot(L,D,'k');
    hold on;
end

a = grayScaleDerivativeMesh(50);
b = grayScaleDerivativeMesh(51);
L = a:1:b;
D = (L-a)/(b-a);
plot(L,D,'k');
grid on;
ylim([0 1.2])
