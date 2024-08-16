clear all;
close all;

noisyLena = imread('../data/simData/noisyLenaGrayScale.jpg');
figure; imagesc(noisyLena)
colormap gray


A = imread('../data/outputPDDODerivative5/rescaledImage/denoisedImage.jpg');
figure, imshow(A, []);
title('Original Image');
figure; histogram(A)

ref = imread('../data/simData/cameraman.png');
figure, imshow(ref, []);
title('Reference Image');
figure; histogram(ref)

BPoly = imhistmatch(A, ref, 'method', 'polynomial');
BUni = imhistmatch(A, ref, 'method', 'uniform');
figure, montage({ref,noisyLena,BPoly, BUni},'Size',[1 4]);


