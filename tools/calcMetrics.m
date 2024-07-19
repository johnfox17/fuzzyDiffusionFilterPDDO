close all;
clear all;
addpath('../data/simData/')
pathFolder = '../data/outputColorImage1/';

noisyLena = single(imread('noisyLena.png'));
lena = imread('Lena.png');



mssim0 = [];
mssim1 = [];
mssim2 = [];
PSNR0 = [];
PSNR1 = [];
PSNR2 = [];

for iImage = 0:131
   
    currentImage = single(imread(pathFolder+"denoisedImage"+string(iImage)+".jpg"));
    lenaChan0 = lena(:,:,1);
    lenaChan1 = lena(:,:,2);
    lenaChan2 = lena(:,:,3);
    currentImageChan0 = imadjustn(currentImage(:,:,1));
    currentImageChan1 = imadjustn(currentImage(:,:,2));
    currentImageChan2 = imadjustn(currentImage(:,:,3));
    [mssim_0, ssim_map_0] = ssim_index(lenaChan0,currentImageChan0 );
    [mssim_1, ssim_map_1] = ssim_index(lenaChan1, currentImageChan1);
    [mssim_2, ssim_map_2] = ssim_index(lenaChan2, currentImageChan2);
    mssim0 = [mssim0, mssim_0];
    mssim1 = [mssim1, mssim_1];
    mssim2 = [mssim2, mssim_2];

    currentPSNR0 = 10.*log10(255^2/(norm(single(lenaChan0(:))-currentImageChan0(:)))^2);
    currentPSNR1 = 10.*log10(255^2/(norm(single(lenaChan1(:))-currentImageChan1(:)))^2);
    currentPSNR2 = 10.*log10(255^2/(norm(single(lenaChan2(:))-currentImageChan2(:)))^2);
    PSNR0 = [PSNR0, currentPSNR0];
    PSNR1 = [PSNR1, currentPSNR1];
    PSNR2 = [PSNR2, currentPSNR2];
end
writematrix(mssim0, pathFolder+"mssim0.csv");
writematrix(mssim1, pathFolder+"mssim1.csv");
writematrix(mssim2, pathFolder+"mssim2.csv");
writematrix(PSNR0, pathFolder+"PSNR0.csv");
writematrix(PSNR1, pathFolder+"PSNR1.csv");
writematrix(PSNR2, pathFolder+"PSNR2.csv");






