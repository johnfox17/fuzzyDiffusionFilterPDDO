clear all;
close all;
addpath('../data/');
%read image
lena  = imread('Lena.png');
%convert to grayscale
lena = rgb2gray(lena);
%create gaussian noise
noisyLena = imnoise(lena,'gaussian', 0,.0025);
%noisyLena = imnoise(lena,'gaussian', 0,.01);
%noisyLena = imnoise(lena,'gaussian');
%noisyLena = awgn(single(lena(:)),10,'measured');
%noisyLena = reshape(noisyLena,[512 512]);
figure; imagesc(lena);
colormap gray
figure; imagesc(noisyLena)
colormap gray
imwrite(noisyLena,'../data/simData/noisyLena.png')

