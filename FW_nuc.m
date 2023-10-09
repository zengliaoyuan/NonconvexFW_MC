function [Xk, iter, fval, history] = FW_nuc(mat_comp_instance, m, n, mu, sigma, options)
% This is the nonconvex Frank-Wolfe method for solving
% min 0.5 * || Proj_Omega(X) - X_obs_vec ||^2
% s.t.  ||X||_* - mu||X||_F <= sigma

% Inputs:
%       mat_comp_instance: struct; include
%           X_obs_vec, irow, jcol (the row index and column index of the observed entries)
%           X_test_vec, irow_test, jcol_test (the row index and column index of the test set entries)
%       options: fields
%           Xinit: the initial point; default set as zero matrix with d=0, U = 0, V = 0, Xinit_obs_vec =0;
%           dsp: the display flag; default set as 0
%           tol: the tolerance for termination; default set as 1e-2
%           maxiter: the limit of iteration numbers; default set as 1000
%           maxtime: the limit of CPU time; default set as 10800 (3h)

% Outputs
%       Xk: struct; store the SVD decomposition (Xk.U, Xk.V, Xk.d) of the kth iterate Xk and
%       the vector of entries at the observed locations Xk_obs_vec

% preparations
X_obs_vec = mat_comp_instance.X_obs_vec;
irow = mat_comp_instance.irow;
jcol = mat_comp_instance.jcol;

X_test_vec = mat_comp_instance.X_test_vec;
irow_test = mat_comp_instance.irow_test;
jcol_test = mat_comp_instance.jcol_test;

if isfield(options, 'dsp')
    dsp = options.dsp;
else
    dsp = 0;
end
if isfield(options, 'tol')
    tol = options.tol;
else
    tol = 1e-4;
end
if isfield(options, 'maxiter')
    maxiter = options.maxiter;
else
    maxiter = 10000;
end
if isfield(options, 'maxtime')
    maxtime = options.maxtime;
else
    maxtime = 10800;
end

if isfield(options, 'isafw')            % using the away step FW by default
    isafw = options.isafw;
else
    isafw = 1;
end

no_obs = length(X_obs_vec);         % number of observed entries
eigifp_init = ones(m+n,1)/sqrt(m+n);          % initial vector for the eigifp solver

c = 1e-4;           % parameter for line search
zeta = 1e5;         % upper bound for stepsizes of away steps
alpha_init = 1;     % initial step size for line search
alpha_fw = alpha_init;

% initialize at 0 by default
if isfield(options, 'Xinit')
    Xk = options.Xk;
else
    Xk = struct();
    Xk.vec = zeros(no_obs, 1);
    Xk.U = zeros(m, 1);
    Xk.V = zeros(n, 1);
    Xk.d = 0;
end
grad_vec = Xk.vec - X_obs_vec;
fval = norm(grad_vec)^2/2;
test_set_err = test_objval(Xk, irow_test, jcol_test, X_test_vec);

if dsp
    if ~isafw
        fprintf('*************** Begin of FW_nuc ***************\n');
    else
        fprintf('*************** Begin of AFW_nuc ***************\n');
    end
    fprintf('%6s| %6s | %6s | %6s | %6s | %6s| %6s| %6s| %8s| %8s\n', ' iter' , 'iter_ls',  'is_aw', 'is_bst', 'rank', 'fold', 'fval',  'Fdiff',  'stepsize', 'fw_gap')
end

% initialize history
history = struct();
history.cputimes = [0];
history.fvals = [fval];
history.test_errs = [test_set_err];
history.ranks = [sum(Xk.d > 1e-6)];
history.fw_gaps_rel = [Inf];
history.bst_recs = [0];
if isafw
    history.aw_recs = [0];
end

iter = 1;
while 1
    iteration_timer = tic;
    
    % the Frank-Wolfe oracle
    Grad = sparse(irow, jcol, grad_vec);
    [utilde_fw, v_fw, zBz] = Lomat(Grad, Xk.U, Xk.V, Xk.d, mu, eigifp_init);
    eigifp_init = [utilde_fw; v_fw];
    u_fw = (2*sigma/zBz) * utilde_fw;
    X_fw_vec = u_fw(irow) .* v_fw(jcol);
    dir_obs_fw = X_fw_vec - Xk.vec;
    dderiv_fw = grad_vec' * dir_obs_fw;
    
    if isafw
        % the away-step oracle
        [u_aw, v_aw, alpha_aw] = awstep_nuc(Grad, sigma, Xk.U,  Xk.V, Xk.d, mu, m, n, zeta);
        X_aw_vec = u_aw(irow) .* v_aw(jcol);
        dir_obs_aw = Xk.vec - X_aw_vec;
        dderiv_aw = grad_vec' * dir_obs_aw;
        
        % choosing a direction
        if dderiv_fw > dderiv_aw && alpha_aw > 1e-5
            is_aw = 1;
            dir_obs_vec = dir_obs_aw;   % direction on Omega
            fw_gap_neg = dderiv_aw;
            alpha_init = alpha_aw;
        else
            is_aw = 0;
            dir_obs_vec  = dir_obs_fw;  % direction on Omega
            fw_gap_neg = dderiv_fw;
        end
        history.aw_recs = [history.aw_recs; is_aw];
    else
        is_aw = 0;
        dir_obs_vec  = dir_obs_fw;   % direction on Omega
        fw_gap_neg = dderiv_fw;
    end
    
    % linesearch 
    iter_ls = 0;
    alpha = alpha_init;
    grad_vec_tilde = grad_vec + alpha*dir_obs_vec;
    ftilde = norm(grad_vec_tilde)^2/2;
    while 1
        if ftilde - fval > c*alpha*fw_gap_neg
            alpha = alpha/2;
            grad_vec_tilde = grad_vec + alpha*dir_obs_vec;
            ftilde = norm(grad_vec_tilde)^2/2;
            iter_ls = iter_ls +1;
        else
            break
        end
    end
    
    % update Xtilde's svd decomposition
    Xk.vec = Xk.vec + alpha*dir_obs_vec;
    Xold = Xk;
    if is_aw
        u = u_aw;
        v = (- alpha)*v_aw;
        Xold.d = (1 + alpha)*Xold.d;
        Xtilde = update_svd_asIF(Xold, u, v);
    else
        u = u_fw;
        v = alpha*v_fw;
        Xold.d = (1 - alpha)*Xold.d;
        Xtilde = update_svd_asIF(Xold, u, v);
    end

    fold = fval;
    % heuristics boosting
    if options.boosting
        cval = norm(Xtilde.d, 1) - mu*norm(Xtilde.d);
        Xk.vec = (sigma/cval)*Xtilde.vec;
        Xk.d = (sigma/cval)*Xtilde.d;
        Xk.U = Xtilde.U;
        Xk.V = Xtilde.V;
        grad_vec = Xk.vec - X_obs_vec;
        fval = norm(grad_vec)^2/2;
        bst_flag = 1;
        if fval > ftilde
            Xk = Xtilde;
            grad_vec = grad_vec_tilde;
            fval = ftilde;
            bst_flag = 0;
        end
    else
        Xk = Xtilde;
        grad_vec = grad_vec_tilde;
        fval = ftilde;
    end
    
    % update alpha_init
    if ~is_aw
        alpha_fw = alpha;
    end
    if (~is_aw) && iter_ls == 0 
        alpha_init = min(max(alpha_fw*2, 1e-8), 1);
    else
        alpha_init = min(max(alpha_fw, 1e-8), 1);
    end
    
    Fdiff = abs(fval - fold)/max(fold,1);
    if dsp
        fprintf('%6d| %6d| %6d| %6d| %6d| %6.4e| %6.4e| %6.3e| %8.3e| %8.3e\n',...
            iter, iter_ls, is_aw, bst_flag, sum(Xk.d > 1e-6), fold, fval, Fdiff, alpha, -fw_gap_neg);
    end
    
    % history update
    history.fvals = [history.fvals; fval];
    test_set_err = test_objval(Xk, irow_test, jcol_test, X_test_vec);
    history.test_errs = [history.test_errs; test_set_err];
    history.ranks = [history.ranks; sum(Xk.d > 1e-6)];
    fw_gap_rel = abs(fw_gap_neg) / max(abs(fold + fw_gap_neg), 1);      % using a similar fomula of this quantity in InFace_Extended_FW
    history.fw_gaps_rel = [history.fw_gaps_rel; fw_gap_rel];
    history.bst_recs = [history.bst_recs; bst_flag];
    iterscost_now = history.cputimes(end) + toc(iteration_timer);
    history.cputimes = [history.cputimes; iterscost_now];
    
    % termination
    if  fw_gap_rel < tol    % terminate when fw_gap is small; 
        fprintf('FW_gap tolerance reached at iteration %d: Fdiff = %6.2e, fw_gap = %6.2e, rank = %d, iterscost_time = %8.4e\n', ...
            iter, Fdiff, -fw_gap_neg, sum(Xk.d > 1e-6), history.cputimes(end));
        break
    end
    
    if  iter >= maxiter
        fprintf('Max_iter reached: iters = %d, Fdiff = %6.2e, fw_gap = %6.2e, rank = %d, iterscost_time = %8.4e \n', ...
            iter, Fdiff,  -fw_gap_neg, sum(Xk.d > 1e-6), history.cputimes(end));
        break
    end
    
    if history.cputimes(end) >= maxtime
        fprintf('Time limit reached at iteration %d: Fdiff = %6.2e, fw_gap = %6.2e, rank = %d, iterscost_time = %8.4e \n', ...
            iter, Fdiff,  -fw_gap_neg, sum(Xk.d > 1e-6),  history.cputimes(end));
        break
    end
    
    iter = iter + 1;
    
end