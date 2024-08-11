clear all;
close all;

pathFolderInit = '../data/output_0.8Mean/';
pathFolderFinal = '../data/output_0.8Mean/';
mmssim = [];
psnr = [];


   
MSSIMInit = table2array(readtable(pathFolderInit+"mssim.csv"));
MSSIMFinal = table2array(readtable(pathFolderFinal+"mssim.csv"));
figure; plot(MSSIMInit,'o'); hold on; plot(MSSIMFinal,'*');
title('MSSIM'); grid on;


PSNRInit = table2array(readtable(pathFolderInit+"PSNR.csv"));
PSNRFinal = table2array(readtable(pathFolderFinal+"PSNR.csv"));
figure; plot(PSNRInit,'o'); hold on; plot(PSNRFinal,'*')
title('PSNR'); grid on;



