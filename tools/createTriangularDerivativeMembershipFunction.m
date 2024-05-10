clear all;
close all;


gradients = {};
a=-255;
b=-212.5;
c=-170;
d=-127.5;
e=-85;
f=-42.5;
g=0;
h=42.5;
i=85;
j=127.5;
k=170;
l=212.5;
m=255;

%D_-6
L = a:1:b;
D = (b-L)/(b-a);
figure; plot(L,D,'o');
hold on;
writematrix([L.',single(D).'],'../data/D_-6.csv');

%D_-5
L_left = a:1:b;
L_right = b:1:c;
D_left = (L_left-a)/(b-a);
D_right = (c- L_right)/(c-b);
D = [D_left,D_right];
L = [L_left,L_right];
plot(L,D,'o');
writematrix([L.',single(D).'],'../data/D_-5.csv');

%D_-4
L_left = b:1:c;
L_right = c:1:d;
D_left = (L_left-b)/(c-b);
D_right = (d- L_right)/(d-c);
D = [D_left,D_right];
L = [L_left,L_right];
plot(L,D,'o');
writematrix([L.',single(D).'],'../data/D_-4.csv');

%D_-3
L_left = c:1:d;
L_right = d:1:e;
D_left = (L_left-c)/(d-c);
D_right = (e- L_right)/(e-d);
D = [D_left,D_right];
L = [L_left,L_right];
plot(L,D,'o');
writematrix([L.',single(D).'],'../data/D_-3.csv');

%D_-2
L_left = d:1:e;
L_right = e:1:f;
D_left = (L_left-d)/(e-d);
D_right = (f- L_right)/(f-e);
D = [D_left,D_right];
L = [L_left,L_right];
plot(L,D,'o');
writematrix([L.',single(D).'],'../data/D_-2.csv');

%D_-1
L_left = e:1:f;
L_right = f:1:g;
D_left = (L_left-e)/(f-e);
D_right = (g- L_right)/(g-f);
D = [D_left,D_right];
L = [L_left,L_right];
plot(L,D,'o');
writematrix([L.',single(D).'],'../data/D_-1.csv');

%D_0
L_left = f:1:g;
L_right = g:1:h;
D_left = (L_left-f)/(g-f);
D_right = (h- L_right)/(h-g);
D = [D_left,D_right];
L = [L_left,L_right];
plot(L,D,'o');
writematrix([L.',single(D).'],'../data/D_0.csv');

%D_1
L_left = g:1:h;
L_right = h:1:i;
D_left = (L_left-g)/(h-g);
D_right = (i- L_right)/(i-h);
D = [D_left,D_right];
L = [L_left,L_right];
plot(L,D,'o');
writematrix([L.',single(D).'],'../data/D_1.csv');

%D_2
L_left = h:1:i;
L_right = i:1:j;
D_left = (L_left-h)/(i-h);
D_right = (j- L_right)/(j-i);
D = [D_left,D_right];
L = [L_left,L_right];
plot(L,D,'o');
writematrix([L.',single(D).'],'../data/D_2.csv');

%D_3
L_left = i:1:j;
L_right = j:1:k;
D_left = (L_left-i)/(j-i);
D_right = (k- L_right)/(k-j);
D = [D_left,D_right];
L = [L_left,L_right];
plot(L,D,'o');
writematrix([L.',single(D).'],'../data/D_3.csv');

%D_4
L_left = j:1:k;
L_right = k:1:l;
D_left = (L_left-j)/(k-j);
D_right = (l- L_right)/(l-k);
D = [D_left,D_right];
L = [L_left,L_right];
plot(L,D,'o');
writematrix([L.',single(D).'],'../data/D_4.csv');

%D_5
L_left = k:1:l;
L_right = l:1:m;
D_left = (L_left-k)/(l-k);
D_right = (m- L_right)/(m-l);
D = [D_left,D_right];
L = [L_left,L_right];
plot(L,D,'o');
writematrix([L.',single(D).'],'../data/D_5.csv');

%D_6
L = l:1:m;
D = (L-l)/(m-l);
plot(L,D,'o');
grid on;
writematrix([L.',single(D).'],'../data/D_6.csv');



legend('D_{-6}','D_{-5}','D_{-4}','D_{-3}','D_{-2}','D_{-1}','D_0','D_1','D_2','D_3','D_4','D_5','D_6')


%Create lookup table
%1-64 DR 0-63
%65-191 GR 64-190
%66-130 WH 191-255
L = 0:1:255; %0 based for python use
grayLevelMembershipTable = [zeros(length(DR(1:64)),1),DR(1:64).';...
    ones(length(GR(65:191)),1),GR(65:191).';...
    2.*ones(length(WH(66:130)),1),WH(66:130).'];
writematrix(single(grayLevelMembershipTable),'../data/triangularMembershipFunction.csv');