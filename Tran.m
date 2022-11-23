%Deterministically calculates a homogenous transform between two frames
%given a set of matched pair of position measurements from point of view of
%each frame

%assume that scaling is 1:1 (coordinate systems both in cm) 

%Solution of the form: 

% T_BA = 
%rxx rxy rxz tx
%ryx ryy ryz  ty
%rzx rzy rzz  tz
%0   0   0    1

%Goal is to minimize the square of the residuals for system of equation
%NOTE: A_m is not invertible when all data lies within the same plane...
%Use pseudo inverse instead
TestDataA = [0.5449, 0.1955, 0.9227;
             0.6862, 0.7202, 0.8004;
             0.8936, 0.7218, 0.2859;
             0.0548, 0.8778, 0.5437;
             0.3037, 0.5824, 0.9848;
             0.0462, 0.0707, 0.7157];
TestDataB = [2.5144, 7.0691, 1.9754;
             2.8292, 7.4454, 2.2224;
             3.3518, 7.3060, 2.1198;
             2.8392, 7.8455, 1.6229;
             2.4901, 7.5449, 1.9518;
             2.4273, 7.1354, 1.4349];
[n,d] = size(TestDataA)
xA = TestDataA(:,1);
yA = TestDataA(:,2);
zA = TestDataA(:,3);
xB = TestDataB(:,1);
yB = TestDataB(:,2);
zB = TestDataB(:,3);
A_m = [sum(xA.^2) sum(xA.*yA) sum(xA.*zA) sum(xA);
       sum(xA.*yA) sum(yA.^2) sum(yA.*zA) sum(yA);
       sum(xA.*zA) sum(yA.*zA) sum(zA.^2) sum(zA);
       sum(xA)     sum(yA)    sum(zA)     n      ];

row1 = pinv(A_m)*[sum(xB.*xA); sum(xB.*yA); sum(xB.*zA); sum(xB)];
row2 = pinv(A_m)*[sum(yB.*xA);sum(yB.*yA);sum(yB.*zA);sum(yB)];
row3 = pinv(A_m)*[sum(zB.*xA); sum(zB.*yA); sum(zB.*zA); sum(zB)];
T_BA = [row1';row2';row3'; 0 0 0 1];

%To calculate point in coordinate system B using a point in coordinate system A, use P_b = T_BA*P_a

%TestDataA = TRUS data 
%TestDataB = da Vinci Data 



