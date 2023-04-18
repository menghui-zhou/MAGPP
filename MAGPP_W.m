

%% update W
function [W, funcVal] = MAGPP_W(X, Y, R, rho1, rho3, rho4, opts)

if nargin < 4
    error('\n Inputs: X, Y, rho1 should be specified!\n');
end

if nargin < 5
    opts = [];
end

% initialize options.
opts = init_opts(opts);

X = multi_transpose(X);
task_num  = length (X);
dimension = size(X{1}, 1);
funcVal = [];


% initialize a starting point
if opts.init==2
    W0 = zeros(dimension, task_num);
elseif opts.init == 0
    W0 = W0_prep;
else
    if isfield(opts,'W0')
        W0=opts.W0;
        if (nnz(size(W0)-[dimension, task_num]))
            error('\n Check the input .W0');
        end
    else
        W0=W0_prep;
    end
end


T =  eye(size(R)) - R; % I - R

bFlag=0; % this flag tests whether the gradient step only changes a little


Wz= W0;
Wz_old = W0;

t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;


while iter < opts.maxIter
    alpha = (t_old - 1) /t;

    %   Ws = (1 + alpha) * Wz - alpha * Wz_old;  % search point / new start point

    Ws = Wz + alpha * (Wz - Wz_old); % search point / new start point

    % compute function value and gradients of the search point
    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval (Ws);


    while true
        Wzp = AutoTR_Projected(Ws - gWs/gamma,  rho3 / gamma, rho4 / gamma);  % gradient descent
        Fzp = funVal_eval(Wzp);

        delta_Wzp = Wzp - Ws;
        r_sum = norm(delta_Wzp, 'fro')^2;


        Fzp_gamma = Fs + sum(sum(delta_Wzp .* gWs))...
            + gamma/2 * r_sum;

        if (r_sum <= 1e-20)
            bFlag = 1; % this shows that, the gradient step makes little improvement
            break;
        end

        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end

    Wz_old = Wz;
    Wz = Wzp;

    funcVal = cat(1, funcVal, Fzp + NonSmooth_eval(Wz, rho3, rho4));

    if (bFlag)
        %         fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end

    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end

    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);

end

W = Wzp;

% private function

% smooth part gradient.
    function [grad_W] = gradVal_eval(W)
        if opts.pFlag
            grad_W = zeros(size(W));
            parfor i = 1:task_num
                grad_W(:, i) = X{i}*(X{i}' * W(:,i)-Y{i});
            end
        else
            grad_W = [];
            for i = 1:task_num
                grad_W = cat(2, grad_W, X{i}*(X{i}' * W(:,i)-Y{i}) );
            end
        end

        grad_W = grad_W + rho1 * 2 * W * T * (T');
    end

% smooth part function value.
    function [funcVal] = funVal_eval (W)
        funcVal = 0;
        if opts.pFlag
            parfor i = 1: task_num
                funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
            end
        else
            for i = 1: task_num
                funcVal = funcVal + 0.5 * norm (Y{i} - X{i}' * W(:, i))^2;
            end
        end
        funcVal = funcVal + rho1 * norm(W*T,'fro')^2 ;
    end



    function [non_smooth_value] = NonSmooth_eval(W, rho_3,  rho_4)
        non_smooth_value = 0;
        for i = 1 : size(W, 1)
            w = W(i, :);
            non_smooth_value = non_smooth_value ...
                + rho_3 * norm(w, 2) + rho_4 * norm(w, 1);
        end
    end


    % compute the projected gradient
    function Wp = AutoTR_Projected(W,  rho3, rho4)
        Wp = zeros(size(W));
        % rowise operator
        for i = 1 : size(W,1)
            w_row = W(i, :);
            w_Lasso = max(0, abs(w_row) - rho4) .* sign(w_row);

            nm = norm(w_Lasso, 2);
            if nm == 0
                w_group_Lasso = 0;
            else
                w_group_Lasso = max(nm - rho3, 0)/nm * w_Lasso;
            end

            Wp(i, :) = w_group_Lasso;
        end

    end


end


