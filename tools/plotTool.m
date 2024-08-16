clear all;
close all;

addpath('../data/simData/')

lena = imread('../data/simData/referenceImageGrayScale.jpg');
% lena = rgb2gray(lena);
figure; imagesc(lena)
colormap gray
figure;histogram(lena)

noisyLena = imread('../data/simData/noisyImageGrayScale.jpg');
figure; imagesc(noisyLena)
colormap gray
figure;histogram(noisyLena)


% noisyImage0 = table2array(readtable('../data/outputPDDODerivative3/coefficients.csv'));
% figure; imagesc(abs(noisyImage0))
% colormap gray
% colorbar
% figure;histogram(noisyImage0)
% % 
% 
% noisyImage0 = table2array(readtable('../data/outputPDDODerivative/gradientCoefficients01.csv'));
% figure; imagesc(abs(noisyImage0))
% colormap gray
% colorbar
% figure;histogram(noisyImage0)


% noisyImage0 = table2array(readtable('../data/output/D01.csv'));
% figure; imagesc(noisyImage0)
% colormap gray
% colorbar
% 
% noisyImage1 = table2array(readtable('../data/outputColorImage7/RHS.csv'));
% figure; imagesc(abs(noisyImage1))
% colormap gray
% colorbar

% noisyImage1 = table2array(readtable('../data/outputColorImage7/gradientCoefficients01.csv'));
% figure; imagesc(abs(noisyImage1))
% colormap gray
% colorbar


% noisyLena = imread('../data/output/2_denoisedImage.jpg');
%noisyLena1 = imread('../data/output/0_denoisedImage.jpg');
% noisyLena = table2array(readtable('../data/outputPDDODerivative8/5_denoisedImage.csv'));
noisyLena1 = table2array(readtable('../data/outputPDDODerivative8/2830_denoisedImage.csv'));

% noisyLena1 = uint8(table2array(readtable('../data/outputPDDODerivative4/15_denoisedImage.csv')));
%imwrite(noisyLena1,"../data/outputPDDODerivative3/rescaledImage/denoisedImage.jpg")
figure; histogram(noisyLena1)
% noisyLena1 = imadjustn(noisyLena1,[0.1 0.9]);
% noisyLena1(noisyLena1>150) =150;
% noisyLena1 = imgaussfilt(noisyLena1,0.5);
% noisyLena1 = imgaussfilt(noisyLena1,0.7);

% noisyLena1 = imadjustn(noisyLena1,[0.025 0.85]);

% n = 2;  
% Idouble = im2double(noisyLena1); 
% avg = mean2(Idouble);
% sigma = std2(Idouble);

% noisyLena1 = imadjustn(noisyLena1,[avg-n*sigma avg+n*sigma],[]);
for i = 1:1
    figure;
    tiledlayout(1,4);
    
    ax1 = nexttile;
    imagesc(lena(:,:,i))
    colormap gray
    colorbar
    title("Channel"+string(i) )
    
    ax2 = nexttile;
    imagesc(noisyLena(:,:,i))
    colormap gray
    colorbar
    title("Channel"+string(i) )
    
    ax3 = nexttile;
    %imagesc(imgaussfilt(noisyLena1(:,:,i),0.4))
    imagesc(noisyLena1(:,:,i))
    title("Channel"+string(i))
    colormap gray;
    colorbar

    ax4 = nexttile;
    imagesc(abs(noisyLena1(:,:,i)-double(noisyLena(:,:,i))))
    title("Channel"+string(i))
    % colormap gray;
    colorbar

    linkaxes([ax1 ax2 ax3 ax4])
    figure; surf(noisyLena1(:,:,i))
end

figure;
tiledlayout(1,4);
ax1 = nexttile;
imagesc(lena)
colorbar
colormap gray
title("Original Lena" )

ax2 = nexttile;
imagesc(noisyLena)
colorbar
colormap gray
title("Noisy Lena" )

ax3 = nexttile;
imagesc(noisyLena1)
colorbar
colormap gray
title("Denoised Lena")

ax4 = nexttile;
% imagesc(imadjustn(imgaussfilt(noisyLena1,0.9)))
% imagesc(imgaussfilt(noisyLena1,0.5))
% imagesc(imgaussfilt(imadjustn(noisyLena1),0.9))
imagesc(imadjustn(noisyLena1,[0.05 0.95]))
% imagesc(imadjustn(noisyLena1))
colorbar
colormap gray
title("Denoised Lena Contrast")
shading interp
linkaxes([ax1 ax2 ax3 ax4])





