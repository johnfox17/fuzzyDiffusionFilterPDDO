clear all;
close all;

pathFolderInit = '../data/outputColorImage1/';
pathFolderFinal = '../data/outputColorImage1/';
mmssim = [];
psnr = [];


   
MSSIMInit0 = table2array(readtable(pathFolderInit+"mssim0.csv"));
MSSIMFinal0 = table2array(readtable(pathFolderFinal+"mssim0.csv"));
figure; plot(MSSIMInit0,'o'); hold on; plot(MSSIMFinal0,'*')
title('MSSIM Channel 0'); grid on;

MSSIMInit1 = table2array(readtable(pathFolderInit+"mssim1.csv"));
MSSIMFinal1 = table2array(readtable(pathFolderFinal+"mssim1.csv"));
figure; plot(MSSIMInit1,'o'); hold on; plot(MSSIMFinal1,'*')
title('MSSIM Channel 1'); grid on;

MSSIMInit2 = table2array(readtable(pathFolderInit+"mssim2.csv"));
MSSIMFinal2 = table2array(readtable(pathFolderFinal+"mssim2.csv"));
figure; plot(MSSIMInit2,'o'); hold on; plot(MSSIMFinal2,'*')
title('MSSIM Channel 2'); grid on;


PSNRInit0 = table2array(readtable(pathFolderInit+"PSNR0.csv"));
PSNRFinal0 = table2array(readtable(pathFolderFinal+"PSNR0.csv"));
figure; plot(PSNRInit0,'o'); hold on; plot(PSNRFinal0,'*')
title('PSNR Channel 0'); grid on;

PSNRInit1 = table2array(readtable(pathFolderInit+"PSNR1.csv"));
PSNRFinal1 = table2array(readtable(pathFolderFinal+"PSNR1.csv"));
figure; plot(PSNRInit1,'o'); hold on; plot(PSNRFinal1,'*')
title('PSNR Channel 1'); grid on;

PSNRInit2 = table2array(readtable(pathFolderInit+"PSNR2.csv"));
PSNRFinal2 = table2array(readtable(pathFolderFinal+"PSNR2.csv"));
figure; plot(PSNRInit2,'o'); hold on; plot(PSNRFinal2,'*')
title('PSNR Channel 2'); grid on;

currentPSNR = table2array(readtable(pathFolder+"PSNR.csv"));
mmssim = [mmssim; currentMSSIM];
psnr = [psnr; currentPSNR];    


figure; hold on;
for iThreshold = 1:length(THRESHOLDS)
   plot(mmssim(iThreshold,:))
end
%ylim([0.51 0.52])
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

