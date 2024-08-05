clear all;
close all;
addpath('../data/simData');
%read image
lena  = imread('cameraman.png');
%noisyLena1  = imread('noisyLena.png');
% [numRows,numCols,numChannels] = size(lena);
% noise = [];
% noisyLena = zeros(numRows,numCols,numChannels);
% for iChan =1:numChannels 
%     for iRow = 1:numRows
%         for iCol = 1:numCols
%             noisyLena(iRow, iCol, iChan) = awgn(double(lena(iRow, iCol, iChan)), 15, 'measured');
%         end
%     end
% end
% noisyLena = uint8(noisyLena);
%convert to grayscale
%lena = rgb2gray(lena);
%create gaussian noise
% noisyLena = imnoise(lena,'gaussian', 0,.04);
%noisyLena = imnoise(lena,'gaussian', 0,.1);

%noisyLena = imnoise(lena,'gaussian');
%noisyLena = awgn(single(lena(:)),10,'measured');
%noisyLena = reshape(noisyLena,[512 512]);
%noisyLena = imnoise(lena,'salt & pepper');
% noisyLena = zeros(512,512,3);
mean = 0;
var_gauss = 0.003;
noisyLena = imnoise(lena,'gaussian',mean,var_gauss);


% figure; imagesc(lena)
% for iChannel = 1:3
%     noise = wgn(512,512,35);
%     figure; surf(noise)
%     % figure; surf(single(lena(:,:,iChannel)))
%     %noisyLena(:,:,iChannel) = single(lena(:,:,iChannel)) + noise;
%     %imnoise(I,'gaussian',mean,var_gauss)
% end
% noisyLena = uint8(noisyLena);


figure; imagesc(lena);
colormap gray
%figure; imagesc(rgb2gray(noisyLena))
figure; imagesc(noisyLena)
colormap gray
imwrite(noisyLena,'../data/simData/noisyCameraman.png')

