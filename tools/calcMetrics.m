% close all;
clear all;
addpath('../data/simData/')
pathFolder = '../data/output2/';

noisyLena = single(imread('noisyLenaGrayScale.jpg'));
% lena = rgb2gray(imread('Lena.png'));
lena = imread('cameraman.png');


mssim = [];
PSNR = [];

for iImage = 0:3
   
    currentImage = single(imread(pathFolder+string(iImage)+"_"+"denoisedImage.jpg"));
    
    % [currentMssim, currentSsim_map] = ssim_index(lena,imadjustn(currentImage) );
   [currentMssim, currentSsim_map] = ssim_index(lena,currentImage );
    mssim = [mssim, currentMssim];
    
    currentPSNR = 10.*log10(255^2/(norm(single(lena(:))-currentImage(:)))^2);
    PSNR = [PSNR, currentPSNR];
    
end
writematrix(mssim, pathFolder+"mssim.csv");
writematrix(PSNR, pathFolder+"PSNR.csv");






