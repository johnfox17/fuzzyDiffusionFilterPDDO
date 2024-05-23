clear all;
% close all;

addpath('../data/simData/')

lena = imread('../data/simData/Lena.png');
lena = rgb2gray(lena);
figure; imagesc(lena)
colormap gray

noisyLena = imread('../data/simData/noisyLena.png');
figure; imagesc(noisyLena)
colormap gray


% DerivativeRule = table2array(readtable("../data/output/DerivativeRule.csv"));
% figure; 
% imagesc(DerivativeRule.')
% colormap gray
% colorbar
% figure;
% histogram(DerivativeRule)
% 
% DerivativeRule2 = table2array(readtable("../data/output/DerivativeRule2.csv"));
% figure; 
% imagesc(DerivativeRule2.')
% colormap gray
% colorbar
% figure;
% histogram(DerivativeRule2)
%THRESHOLDS = [0.01, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5];




THRESHOLDS = 0.01;
%timeSteps = ["0","10","50","100","150","200","300"];
timeSteps = ["102"];
for iThreshold = 1:length(THRESHOLDS)
    pathFolder = "../data/output3/threshold_"+string(THRESHOLDS(iThreshold))+"/";
    %pathFolder = "../data/output/threshold_"+string(THRESHOLDS(iThreshold))+"/";
    for iTimeStep = 1:length(timeSteps)
        denoisedImage = table2array(readtable(pathFolder+"denoisedImage"+timeSteps(iTimeStep)+".csv"));
        gradient = table2array(readtable(pathFolder+"gradient"+timeSteps(iTimeStep)+".csv"));
        gradient = reshape(gradient, [512 512]).';
        RHS = table2array(readtable(pathFolder+"RHS"+timeSteps(iTimeStep)+".csv"));
        localSmoothness = table2array(readtable(pathFolder+"localSmoothness"+timeSteps(iTimeStep)+".csv"));
        
        figure;
        tiledlayout(2,2);
    
        ax1 = nexttile;
        imagesc(denoisedImage)
        colormap gray
        colorbar
        title("Lena")
    
        ax2 = nexttile;
        imagesc(gradient)
        title('Gradient ')
        colormap gray;
        colorbar
    
        ax3 = nexttile;
        imagesc(RHS)
        title('Right Hand Side')
        colormap gray;
        colorbar
        
        ax4 = nexttile;
        imagesc(localSmoothness)
        title('Local Smoothness')
        colormap gray;
        colorbar
        
        linkaxes([ax1 ax2 ax3 ax4])
        figure;
        surf(denoisedImage)
    end
end







