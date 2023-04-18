% Objective Function
% argmin_R {0.5\|WR-W\|_F^2 + rho \|R\|_1}
% rho = rho2/rho1/2;

%% Code starts here
function [R, funcVal] = MAGPP_R(W, rho1, rho2, S, opts)

if nargin < 4
    error('\n Inputs: X, Y, rho1 should be specified!\n');
end

if nargin < 5
    opts = [];
end
% initialize options.
opts = init_opts(opts);


% Objective Function
% argmin_R {0.5\|WR-W\|_F^2 + rho \|R\|_1}
rho = rho2/rho1/2;


dimension = size(W, 2);
funcVal = [];

% initialize a starting point
if opts.init==2
    R0 = zeros(dimension, dimension);
elseif opts.init == 0
    R0 = R0_prep;
else
    if isfield(opts,'W0')
        R0=opts.R0;
        if (nnz(size(R0)-[dimension, dimension]))
            error('\n Check the input .R0');
        end
    else
        R0=W0_prep;
    end
end

WtW =  W' * W; % I - R

bFlag=0; % this flag tests whether the gradient step only changes a little


Rz= R0;
Rz_old = R0;

t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;


while iter < opts.maxIter
    alpha = (t_old - 1) /t;

    %   Ws = (1 + alpha) * Wz - alpha * Wz_old;  % search point / new start point

    Rs = Rz + alpha * (Rz - Rz_old); % search point / new start point

    % compute function value and gradients of the search point
    gRs  = gradVal_eval(Rs);
    Fs   = funVal_eval (Rs);


    while true
        Rzp = R_Projected_Gradient(Rs - gRs/gamma, rho/gamma);  % gradient descent
        Fzp = funVal_eval(Rzp);

        delta_Rzp = Rzp - Rs;
        r_sum = norm(delta_Rzp, 'fro')^2;


        Fzp_gamma = Fs + sum(sum(delta_Rzp .* gRs))...
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

    Rz_old = Rz;
    Rz = Rzp;

    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Rz, rho));

    if (bFlag)
%         fprintf(['\n The program terminates as the ' ...
%             'gradient step changes the solution very small.']);
        break;
    end

    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter >= 2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter >= 2
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
            if iter >= opts.maxIter
                break;
            end
    end

    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);

end

R = Rzp;


% private function
    function [R] = R_Projected_Gradient(R, rho)
        R = max(0, abs(R) - rho * S) .* sign(R);
    end

% smooth part gradient.
    function [grad_R] = gradVal_eval(R)
        grad_R = WtW * (R - eye(size(R)));
    end

% smooth part function value.
    function [funcVal] = funVal_eval(R)
        funcVal = 0.5 * norm(W*R-W, 'fro')^2;
    end

% nonsmooth part function value
    function [non_smooth_value] = nonsmooth_eval(R, rho)
        non_smooth_value = 0;
        R = R .* S;
        for i = 1 : size(R, 2)
            w = R(:, i);
            non_smooth_value = non_smooth_value + rho * norm(w, 1);
        end
    end
end