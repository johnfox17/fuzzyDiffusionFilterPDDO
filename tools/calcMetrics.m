close all;
clear all;
addpath('../data/simData/')


noisyLena = single(imread('noisyLena.png'));
lena = imread('Lena.png');
lena = rgb2gray(lena);
THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5];

for iThreshold = 1:length(THRESHOLDS)
    pathFolder = "../data/output/threshold_"+string(THRESHOLDS(iThreshold))+"/";
    mssim = [];
    PSNR = [];
    for iImage = 0:300
       
        currentImage = table2array(readtable(pathFolder+"denoisedImage"+string(iImage)+".csv"));
        [mssim_0, ssim_map_0] = ssim_index(lena, currentImage);
        mssim = [mssim, mssim_0];
    
        currentPSNR = 10.*log10(255^2/(norm(single(lena(:))-currentImage(:)))^2);
        PSNR = [PSNR, currentPSNR];
    end
    writematrix(mssim, pathFolder+"mssim.csv");
    writematrix(PSNR, pathFolder+"PSNR.csv");
end





