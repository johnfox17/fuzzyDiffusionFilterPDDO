close all;
clear all;
addpath('../data/simData/')
addpath('../data/output2/threshold_0_1/')

noisyLena = single(imread('noisyLena.png'));
lena = imread('Lena.png');
lena = rgb2gray(lena);
mssim = [];
PSNR = [];
for iImage = 0:121
    currentImage = table2array(readtable("denoisedImage"+string(iImage)+".csv"));
    [mssim_0, ssim_map_0] = ssim_index(lena, currentImage);
    mssim = [mssim, mssim_0];

    currentPSNR = 10.*log10(255^2/(norm(single(lena(:))-currentImage(:)))^2);
    PSNR = [PSNR, currentPSNR];
end

writematrix(mssim, '../data/output2/threshold_0_1/mssim.csv');
writematrix(PSNR, '../data/output2/threshold_0_1/PSNR.csv');




