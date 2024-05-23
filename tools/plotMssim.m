clear all;
close all;

THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5];
%THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.09, 0.1, 0.15, 0.2];
THRESHOLDS = [0.01];
mmssim = [];
psnr = [];

for iThreshold = 1:length(THRESHOLDS)
    pathFolder = "../data/output3/threshold_"+string(THRESHOLDS(iThreshold))+"/";
    currentMSSIM = table2array(readtable(pathFolder+"mssim.csv"));
    currentPSNR = table2array(readtable(pathFolder+"PSNR.csv"));
    mmssim = [mmssim; currentMSSIM];
    psnr = [psnr; currentPSNR];    
end

figure; hold on;
for iThreshold = 1:length(THRESHOLDS)
   plot(mmssim(iThreshold,:))
end
ylim([0 0.6])
grid on;
%legend('0.01', '0.02', '0.03', '0.05', '0.08', '0.09', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.5');
%legend('0.01', '0.02', '0.03', '0.05', '0.08', '0.09', '0.1', '0.15', '0.2');

figure; hold on;
for iThreshold = 1:length(THRESHOLDS)
    plot(psnr(iThreshold,:))
end
grid on;
%legend('0.01', '0.02', '0.03', '0.05', '0.08', '0.09', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.5');
%legend('0.01', '0.02', '0.03', '0.05', '0.08', '0.09', '0.1', '0.15', '0.2');

