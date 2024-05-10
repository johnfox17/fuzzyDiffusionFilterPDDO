clear all;
close all;


mssim_0_05_1 = table2array(readtable("../data/output/threshold_0_1/mssim.csv"));
mssim_0_05_2 = table2array(readtable("../data/output2/threshold_0_1/mssim.csv"));
figure; 
plot(mssim_0_05_1, '-o')
hold on;
plot(mssim_0_05_2, '-*')











mssim_0_05 = table2array(readtable("../data/threshold_0_05/mssim.csv"));
mssim_0_08 = table2array(readtable("../data/threshold_0_08/mssim.csv"));
mssim_0_09 = table2array(readtable("../data/threshold_0_09/mssim.csv"));
mssim_0_1 = table2array(readtable("../data/threshold_0_1/mssim.csv"));
mssim_0_2 = table2array(readtable("../data/threshold_0_2/mssim.csv"));
mssim_0_3 = table2array(readtable("../data/threshold_0_3/mssim.csv"));


figure; 
plot(mssim_0_05, '-o')
hold on;
plot(mssim_0_08,'-.')
plot(mssim_0_09,'-.')
plot(mssim_0_1,'-o')
plot(mssim_0_2,'-*')
plot(mssim_0_3,'-^')
grid on;
legend('mssim_0_05','mssim_0_08','mssim_0_09','mssim_0_1','mssim_0_2','mssim_0_3')



psnr_0_05 = table2array(readtable("../data/threshold_0_05/PSNR.csv"));
psnr_0_08 = table2array(readtable("../data/threshold_0_08/PSNR.csv"));
psnr_0_09 = table2array(readtable("../data/threshold_0_09/PSNR.csv"));
psnr_0_1 = table2array(readtable("../data/threshold_0_1/PSNR.csv"));
psnr_0_2 = table2array(readtable("../data/threshold_0_2/PSNR.csv"));
psnr_0_3 = table2array(readtable("../data/threshold_0_3/PSNR.csv"));


figure; 
plot(psnr_0_05, '-o')
hold on;
plot(psnr_0_08,'-.')
plot(psnr_0_09,'-.')
plot(psnr_0_1,'-o')
plot(psnr_0_2,'-*')
plot(psnr_0_3,'-^')
grid on;
legend('psnr_0_05','psnr_0_08','psnr_0_09','psnr_0_1','psnr_0_2','psnr_0_3')
