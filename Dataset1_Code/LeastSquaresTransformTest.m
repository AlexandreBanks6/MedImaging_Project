x = [1 2 3;
     5 1 7; 
     8 9 1;
     12 13 4;
     7 8 10];
x = [x ones(5,1)]';
T = [1 0 0 1
     0 1 0 1
     0 0 1 1 
     0 0 0 1];
y = T*x;


Transform = LeastSquaresNumericalTransform(x, y)

y1 = Transform*x
y1_res = y1'; y1_res(:,4) = []; 
x_res = x'; x_res(:,4) = []; 
err = rmse(y1_res, x_res)
total_error = (err(1)^2 + err(2)^2+err(3)^2)^0.5
