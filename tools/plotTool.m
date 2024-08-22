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


% noisyImage0 = table2array(readtable('../data/output/similarity.csv'));
% figure; imagesc(noisyImage0)
% colormap gray
% colorbar
% figure;histogram(noisyImage0)
 
fuzzySimilarityImage0 = table2array(readtable('../data/output/0_fuzzySimilarityImage.csv')); 
noisyLena0 = table2array(readtable('../data/output/0_denoisedImage.csv'));
figure; histogram(noisyLena0)

figure;
tiledlayout(2,2);

ax1 = nexttile;
imagesc(lena)
colormap gray
colorbar
title("Original Image" )

ax2 = nexttile;
imagesc(noisyLena)
colormap gray
colorbar
title("Noisy Image" )

ax3 = nexttile;
imagesc(imgaussfilt(noisyLena0,0.6))
title("Denoised Image 0")
colormap gray;
colorbar

ax4 = nexttile;
imagesc(fuzzySimilarityImage0)
title("Fuzzy Similarity Image 0")
colormap gray;
colorbar

linkaxes([ax1 ax2 ax3 ax4])
figure; surf(noisyLena0)



fuzzySimilarityImage1 = table2array(readtable('../data/output/100_fuzzySimilarityImage.csv')); 
noisyLena1 = table2array(readtable('../data/output/100_denoisedImage.csv'));
figure; histogram(noisyLena1)

figure;
tiledlayout(2,2);

ax5 = nexttile;
imagesc(lena)
colormap gray
colorbar
title("Original Image" )

ax6 = nexttile;
imagesc(noisyLena)
colormap gray
colorbar
title("Noisy Image" )

ax7 = nexttile;
imagesc(imgaussfilt(noisyLena1,0.6))
title("Denoised Image 1")
colormap gray;
colorbar

ax8 = nexttile;
imagesc(fuzzySimilarityImage1)
title("Fuzzy Similarity Image 1")
colormap gray;
colorbar

linkaxes([ax5 ax6 ax7 ax8])
figure; surf(noisyLena1)



figure;
tiledlayout(2,2);
ax9 = nexttile;
imagesc(imgaussfilt(noisyLena0,0.6))
title("Denoised Image 0")
colormap gray;
colorbar

ax10 = nexttile;
imagesc(fuzzySimilarityImage0)
title("Fuzzy Similarity Image 0")
colormap gray;
colorbar

ax11 = nexttile;
imagesc(imgaussfilt(noisyLena1,0.6))
title("Denoised Image 1")
colormap gray;
colorbar

ax12 = nexttile;
imagesc(fuzzySimilarityImage1)
title("Fuzzy Similarity Image 1")
colormap gray;
colorbar

linkaxes([ax9 ax10 ax11 ax12])


figure;
imagesc(noisyLena0-noisyLena1)
colormap gray;
colorbar

figure;
imagesc(abs(fuzzySimilarityImage0-fuzzySimilarityImage1))
colormap gray;
colorbar
