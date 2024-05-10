clear all;
close all;



%DR
a=1;
b=127;
L = a:1:b;
DR = (b-L)/(b-a);
figure; plot(L,DR,'*');
hold on;


%GR
a=1;
b=127;
c=256;
L_GR_left = a:1:b;
L_GR_right = b:1:c;
GR_left = (L_GR_left-a)/(b-a);
GR_right = (c- L_GR_right)/(c-b);
GR = [GR_left,GR_right(1,2:end)];
L = [L_GR_left,L_GR_right(1,2:end)];
plot(L,GR,'o');

%WH
a=127;
b=256;
L = a:1:b;
WH = (L-a)/(b-a);
plot(L,WH,'*');
grid on;

%Create lookup table
%1-64 DR 0-63
%65-191 GR 64-190
%66-130 WH 191-255
L = 0:1:255; %0 based for python use
grayLevelMembershipTable = [zeros(length(DR(1:64)),1),DR(1:64).';...
    ones(length(GR(65:191)),1),GR(65:191).';...
    2.*ones(length(WH(66:130)),1),WH(66:130).'];
writematrix(single(grayLevelMembershipTable),'../data/triangularMembershipFunction.csv');