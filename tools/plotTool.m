clear all;
close all;

addpath('../data/output/threshold_0.01')
lena = imread('../data/simData/Lena.png');
lena = rgb2gray(lena);

figure; imagesc(lena)
colormap gray

denoisedImage0 = table2array(readtable("denoisedImage0.csv"));
figure; imagesc(denoisedImage0)
title('denoisedImage0')
colormap gray;
colorbar
figure; surf(denoisedImage0)
% 
% g0 = table2array(readtable("g0.csv"));
% g0 = reshape(g0, [512 512]).';
% figure; imagesc(g0)
% title('g0')
% colormap gray;
% figure; surf(g0)
% 
% 
% RHS0 = table2array(readtable("RHS0.csv"));
% figure; imagesc(RHS0)
% title('RHS0')
% colormap gray;
% figure; surf(RHS0)
% 
% 
% localSmoothness0 = table2array(readtable("localSmoothness0.csv"));
% figure; imagesc(localSmoothness0)
% colormap gray;
% title('localSmoothness0')
% figure; surf(localSmoothness0)


numInteration = "150";
denoisedImage1 = table2array(readtable("denoisedImage"+numInteration+".csv"));
% denoisedImage1(denoisedImage1>=210) = 210;
% denoisedImage1(denoisedImage1<=-210) = -210;
%denoisedImage1 = movmean(denoisedImage1, 5);
figure; imagesc(denoisedImage1)
colormap gray;
colorbar
title('denoisedImage1')
figure; surf(denoisedImage1)

g1 = table2array(readtable("g"+ numInteration +".csv"));
g1 = reshape(g1, [512 512]).';
figure; imagesc(g1)
colormap gray;
title('g1')
figure; surf(g1)


RHS1 = table2array(readtable("RHS"+ numInteration +".csv"));
figure; imagesc(RHS1)
colormap gray;
title('RHS1')
figure; surf(RHS1)


localSmoothness1 = table2array(readtable("localSmoothness" + numInteration +".csv"));
figure; imagesc(localSmoothness1)
colormap gray;
title('localSmoothness1')
figure; surf(localSmoothness1)





