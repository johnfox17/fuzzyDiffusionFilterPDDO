clear all;
close all;
addpath('../data/simData');
%read image
lena  = imread('Lena.png');
[numRows,numCols,numChannels] = size(lena);
noise = [];
noisyLena = zeros(numRows,numCols,numChannels);
for iChan =1:numChannels 
    for iRow = 1:numRows
        for iCol = 1:numCols
            noisyLena(iRow, iCol, iChan) = awgn(double(lena(iRow, iCol, iChan)), 1, 'measured');
        end
    end
end
noisyLena = uint8(noisyLena);

%convert to grayscale
%lena = rgb2gray(lena);
%create gaussian noise
%noisyLena = imnoise(lena,'gaussian', 0,.0025);
%noisyLena = imnoise(lena,'gaussian', 0,.1);

%noisyLena = imnoise(lena,'gaussian');
%noisyLena = awgn(single(lena(:)),10,'measured');
%noisyLena = reshape(noisyLena,[512 512]);
%noisyLena = imnoise(lena,'salt & pepper');
figure; imagesc(lena);
colormap gray
figure; imagesc(noisyLena)
colormap gray
imwrite(noisyLena,'../data/simData/noisyLena.png')

