function [T_BA] = LeastSquaresNumericalTransform(TestDataA,TestDataB)
%LeastSquaresNumericalTransform Numerically computes the transform between
%two 3D coordinate systems. TestDataA and B each contain points in their
%respective coordinate systems that were at the same global coordinates as
%each other. Returns the transformation matrix that converts a point from A
%into B
%   Detailed proof found in
%   https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8271986

[n,d] = size(TestDataA);
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
%P_a must be of the form: [xA
%                          yA
%                          zA
%                          1]



end