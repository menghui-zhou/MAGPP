


%% Code starts here
function [W, R, out_list, funcVal] = MAGPP(X, Y, lambda1, lambda2, lambda3, lambda4, lambda5,  Auto_opts)
% Automatic Temporal Smoothness in Multi-task Learning

% Objective Function
% argmin_{W, R} { sum_i^m (0.5 * norm (Y{i} - X{i}' * W(:, i))^2)
%            + rho1 * \|W - WR\|_F^2 + rho2 * \|R .* S\|_1 + rho3 *
%            \|W^T\|_{1,2} + rho * 4 \|W^T\|_{1,1}

% initialization
dim = size(X{1}, 2);
task_num = length(X);


% construct weighted temporal relation matrix
S = (lambda5-1) * eye(task_num) + ones(task_num);

% Auto_opts = [];
% if the property is not defined, define it.
if ~isfield(Auto_opts, 'W_tol')
    Auto_opts.W_tol= 1e-8;
end
if ~isfield(Auto_opts, 'R_tol')
    Auto_opts.R_tol= 1e-8;
end
if ~isfield(Auto_opts, 'maxIter')
    Auto_opts.maxIter = 1000;
end
if ~isfield(Auto_opts, 'R_ini')
    Auto_opts.maxIter = 0; % default: 0 initialization
end


Auto_opts = init_opts(Auto_opts);
Auto_opts.tol = 1e-4;


W = zeros(dim, task_num);

if Auto_opts.R_ini == 0
    R = zeros(task_num, task_num);
elseif Auto_opts.R_ini == 1
    % Initialization with Guassian Kernel
    R = zeros(task_num, task_num);

    for col = 1 : task_num
        sum  = 0;
        for row = 1 : task_num
            if row == col
                R(row,col) = 0;
            else
                R(row,col) = exp(-abs(row-col));
                sum = sum + exp(-abs(row-col));
            end
        end
        R(:, col) = R(:, col) / sum;
    end
elseif Auto_opts.R_ini == 2
    % I - R = temporal smoothness
    R = zeros(task_num,task_num);
    R(task_num+1:(task_num+1):end) = 1;
    R(2:(task_num+1):end) = 1;
end


W_opts = [];
R_opts = [];

% W_change = inf;
% R_change = inf;

out_list = [];
W_change_list = [];
R_change_list = [];
funcVal = [];

num_it = 1;
while num_it <= Auto_opts.maxIter

    % fix R, update W
    W_old = W;
    [W, W_funcVal] = MAGPP_W(X, Y, R, lambda1, lambda3, lambda4, W_opts); %#ok<ASGLU> 
    W_change = norm(W_old - W, 'fro')^2 / norm(W_old, 'fro')^2;
    W_change_list = cat(1, W_change_list, W_change);

    % fix W, update R
    R_old = R;
    [R, R_funcVal] = MAGPP_R(W, lambda1, lambda2, S, R_opts); %#ok<ASGLU> 
    R_change = norm(R_old - R, 'fro')^2 / norm(R_old, 'fro')^2;
    R_change_list = cat(1, R_change_list, R_change);
       


    funcVal = cat(1, funcVal, Get_funcVal(W,R));
 
    if num_it >= 2
        func_relative_change = abs(funcVal(end) -funcVal(end-1)) / funcVal(end-1);

        if mod(num_it, 10 ) == 0
            fprintf('The %d Iteration \n', num_it);
        end

        if num_it == Auto_opts.maxIter - 1
            fprintf('Max Iteration! Relative change of function value is %.11f\n', func_relative_change );
        end


%         if W_change <= Auto_opts.W_tol || R_change <= Auto_opts.R_tol
%             fprintf('The Relative change of W and R: %f     %f\n', W_change, R_change);
%             fprintf('Relative change of function value is %.11f\n', func_relative_change );
%             fprintf('The iteration number is %d\n\n', num_it );
%             break;
%         end


        if func_relative_change < Auto_opts.tol
            fprintf('Relative change of function value is %.11f\n', func_relative_change );
            fprintf('The Relative change of W and R: %f     %f\n', W_change, R_change);
            fprintf('The iteration number is %d\n\n', num_it );
            break;
        end
    end

    num_it = num_it + 1;
end



out_list.funcVal = funcVal;
out_list.W_change = W_change_list;
out_list.R_change = R_change_list;

% private function value
    function funcVal = Get_funcVal(W, R)
        % L(W)
        LW = 0;
        if Auto_opts.pFlag
            parfor i = 1: task_num
                LW = LW + 0.5 * norm (Y{i} - X{i} * W(:, i))^2;
            end
        else
            for i = 1: task_num
                LW = LW + 0.5 * norm (Y{i} - X{i} * W(:, i))^2;
            end
        end

        % \lambda1 \| W- WR\|_F^2 
        FirstTerm = lambda1 * norm(W - W*R, 'fro')^2;

        % \lambda2 \| R .* S \|_1
        SecondTerm = 0;
        R_coe = R .* S;
        if Auto_opts.pFlag
            parfor i = 1 : size(R, 2)
                SecondTerm = SecondTerm + lambda2 * norm(R_coe(:, i), 1);
            end
        else
            for i = 1 : size(R, 2)
                SecondTerm = SecondTerm + lambda2 * norm(R_coe(:, i), 1);
            end
        end


        % \lambda3 \|W^T\|_{1,2} + \lambda4 \|W^T\|_{1,1}
        ThirdTerm = 0;
        if Auto_opts.pFlag
            parfor i = 1 : size(R, 1)
                ThirdTerm = ThirdTerm + lambda3 * norm(W(i, :), 2) + lambda4 * norm(W(i,:), 1);
            end
        else
            for i = 1 : size(R, 2)
                ThirdTerm = ThirdTerm + lambda3 * norm(W(i, :), 2) + lambda4 * norm(W(i,:), 1);
            end
        end

        funcVal = LW + FirstTerm + SecondTerm + ThirdTerm;

    end

end

