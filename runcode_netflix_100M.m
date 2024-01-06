function [historyIF_0Inf,history_fw_bst,history_afw_bst,scale]=runcode_netflix_100M(sigma,mu,time_limit)

%% Inputs:
%   mu: the parameter of the DC regularizer in the constraint
%   sigma: the right hand in the constraint; determined by the file find_delta_asIF.m, which is the same as the 
%               cross-validation code for learning delta in the InFaceExtendedFW_MatrixCompletion solver
%   time_limit: the maximal computational time allowed

%% download the netflix zip file;
dir_exist = exist('netflix', 'dir');  % check the existence of the folder netflix
dir_exist_7 = dir_exist - 7;
if (dir_exist_7)
    if ~isfile('netflix.tar.gz')    % check the existence of the file netflix.tar.gz
        url = 'https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz';
        filename = 'netflix.tar.gz';
        options = weboptions('Timeout',Inf);
        websave(filename, url, options);
        clear options
        fprintf('netflix.tar.gz downloaded. \n')
    end 
        untar('netflix.tar.gz', 'netflix')
        cd netflix/download
        untar('training_set.tar', 'data');
        cd ..
        cd ..
        fprintf('netflix.tar.gz unzipped. \n');
end

%%    read rating.dat; 
%%%   This data reading part is from read_movielens_10M.m file of InFaceExtendedFW-MatrixCompletion

Mcell=cell(1,17770);
for i = 1: 17770
    filename = ['netflix/download/data/training_set/mv_' num2str(i,'%07d')  '.txt'];
    M_temp = readmatrix(filename);
    M_temp(:,3) = i;
    Mcell{1,i} = M_temp;
end
M = vertcat(Mcell{:});

fprintf('Netflix data reading completed. \n')

irow = M(:, 1); % users
jcol = M(:, 3); % movies

%%% fix irow, jcol
[unique_rows, irow_ind, unique_rows_ind] = unique(irow);
[unique_cols, jcol_ind, unique_cols_ind] = unique(jcol);

nrow = length(unique_rows);
ncol = length(unique_cols);

irow = (1:nrow)';
irow = irow(unique_rows_ind);
jcol = (1:ncol)';
jcol = jcol(unique_cols_ind);

Omega = sub2ind([nrow,ncol],irow,jcol);

Xobs_vec = M(:, 2);   % ratings
no_obs = length(Xobs_vec);

%%% center the data
centerMatRows = [1:no_obs, 1:no_obs]';
centerMatCols = [irow; nrow + jcol];
centerMat = sparse(centerMatRows, centerMatCols, ones(2*no_obs, 1));

alphabeta = lsqr(centerMat, Xobs_vec, 10^-6, 1000);
alpha = alphabeta(1:nrow);
beta = alphabeta(nrow+1:nrow+ncol);

Xobs_vec = Xobs_vec - alpha(irow) - beta(jcol);
scale = norm(Xobs_vec, 2);
Xobs_vec = Xobs_vec/norm(Xobs_vec, 2);

save netflix_100M Omega irow jcol  Xobs_vec;  

fprintf('The user/movie data processed. \n')

%% add pathes for running InFaceExtendedFW
addpath('.\InFaceExtendedFW-MatrixCompletion\experiments')
addpath('.\InFaceExtendedFW-MatrixCompletion\experiments\paper-experiments')
addpath('.\InFaceExtendedFW-MatrixCompletion\solver')


%%  Split the dataset into training and testting sets; 
%%% This part is from movielens10M_run.m of InFaceExtendedFW-MatrixCompletion
rng(345);
warning('error', 'MATLAB:eigs:NoEigsConverged');

load('netflix_100M.mat');

[train_data, test_data] = split_matcomp_instance(Omega, irow, jcol, Xobs_vec, 0.7);

mat_comp_instance = struct();
mat_comp_instance.X_obs_vec = train_data.Xobs_vec;
mat_comp_instance.irow = train_data.irow;
mat_comp_instance.jcol = train_data.jcol;

mat_comp_instance.X_test_vec = test_data.Xobs_vec;
mat_comp_instance.irow_test = test_data.irow;
mat_comp_instance.jcol_test = test_data.jcol;

%% value of sigma in the models; named as delta in Freund's paper and codes
mat_comp_instance.delta = sigma;       

%% The InFace direction method 
options = struct();
options.verbose = 1;
options.rel_opt_TOL = -Inf;
options.abs_opt_TOL = -Inf;
options.bound_slack = 10^-6;
options.time_limit = time_limit;
options.max_iter = 40000;
options.prox_grad_tol = 10^-5;

options.gamma_1 = 0;
options.gamma_2 = Inf;
options.last_toward = 0;
options.rank_peak = 0;

options.pre_start_full = 0;

options.test_error_basic = 1;

options.svd_options.large_type = 'eigs';
options.svd_options.maxiter = 5000;
options.svd_options.tol = 10^-8;
options.svd_options.vector_stopping = 0;
options.svd_options.svd_test = 0;

options.alg_type = 'InFace';
fprintf('Start of the InFaceExtendedFW method.\n')
tstart1 = tic;
[final_solnIF_0Inf, historyIF_0Inf] = InFace_Extended_FW_sparse(mat_comp_instance, @Away_step_standard_sparse, @update_svd, options);
t_IF_0Inf = toc(tstart1);


a = clock;
datname = ['InFaceResults_netflix_100M'  '_'  'timelimit' num2str(time_limit) '_date' date '-' int2str(a(4)) '-' int2str(a(5)) '.mat'];
save(datname, 'final_solnIF_0Inf', 'historyIF_0Inf', 't_IF_0Inf');


%% The nonconvex FW and AFW method
clear options
m = 480189;
n = 17770;


options = struct();
options.tol = 1e-6;
options.maxiter = 40000;
options.maxtime = time_limit;
options.dsp = 1;

%%% The boosting FW method
options.isafw = 0;
options.boosting = 1;
fprintf('Start of the FW-boosting method.\n')
tstart2 = tic;
[Xk_fw_bst, iter_fw_bst, fval_fw_bst, history_fw_bst] = FW_nuc(mat_comp_instance, m, n, mu, sigma, options);
t_fw_bst = toc(tstart2);

%%% The away-step FW method with boosting
options.isafw = 1;
options.boosting = 1;
fprintf('Start of the AFW-boosting method.\n')
tstart3 = tic;
[Xk_afw_bst, iter_afw_bst, fval_afw_bst, history_afw_bst] = FW_nuc(mat_comp_instance, m, n, mu, sigma, options);
t_afw_bst = toc(tstart3);


a = clock;
datname = ['FWandAFWResults_netflix_100M_mu'   num2str(mu*100) '_'  'timelimit' num2str(time_limit) '_date' date '-' int2str(a(4)) '-' int2str(a(5)) '.mat'];
save(datname, 'Xk_fw_bst',  'Xk_afw_bst', 'iter_fw_bst', 'iter_afw_bst', 'fval_fw_bst', 'fval_afw_bst', 'history_fw_bst', 'history_afw_bst', 't_fw_bst', 't_afw_bst','scale');


%% plot sum squared error on training set and testing set
iterscosts_fw = history_fw_bst.cputimes;
iterscosts_afw = history_afw_bst.cputimes;
iterscosts_IF = historyIF_0Inf.cputimes(1:historyIF_0Inf.num_iters);

fvals_fw = history_fw_bst.fvals;
fvals_afw = history_afw_bst.fvals;
fvals_IF = historyIF_0Inf.objvals(1:historyIF_0Inf.num_iters);

test_errs_fw = history_fw_bst.test_errs;
test_errs_afw = history_afw_bst.test_errs;
test_errs_IF = historyIF_0Inf.test_set_errors(1:historyIF_0Inf.num_iters);

fig1 = figure;
plot(iterscosts_fw, fvals_fw, '-', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2); hold on
plot(iterscosts_afw, fvals_afw, '-', 'Color', [174 221 129]/255,'LineWidth', 2); hold on
plot(iterscosts_IF, fvals_IF, '-', 'Color', [146 172 209]/255, 'LineWidth', 2); 
legend('trainErr-FW', 'trainErr-AFW', 'trainErr-IF');
y_max = max([max(fvals_fw), max(fvals_afw), max(fvals_IF)]);
y_min = min([min(fvals_fw), min(fvals_afw), min(fvals_IF)]);
yaxis_up = y_max + (y_max - y_min)/100;
yaxis_bottom = y_min - (y_max - y_min)/100;
axis([0 options.maxtime yaxis_bottom yaxis_up])
xlabel('time(s)');
ylabel('Train Error')
a = clock;
figname = ['netflix_100M-trainingErrs-mu' num2str(mu*100)  '-' date '-' num2str(a(4)) '-'  num2str(a(5))];
savefig(fig1, figname, 'Figures');

fig2 = figure;
plot(iterscosts_fw, test_errs_fw, '-', 'Color', [147, 147, 145]/255,'LineWidth', 2); hold on
plot(iterscosts_afw, test_errs_afw, '-', 'Color', [238,130,238]/255, 'LineWidth', 2); hold on
plot(iterscosts_IF, test_errs_IF, '-', 'Color', [226, 17, 0]/255, 'LineWidth', 2); hold off
legend( 'testErr-FW', 'testErr-AFW', 'testErr-IF');
y_max = max([max(test_errs_fw), max(test_errs_afw), max(test_errs_IF)]);
y_min = min([min(test_errs_fw), min(test_errs_afw), min(test_errs_IF)]);
yaxis_up = y_max + (y_max - y_min)/100;
yaxis_bottom = y_min - (y_max - y_min)/100;
axis([0 options.maxtime yaxis_bottom yaxis_up])
xlabel('time(s)');
ylabel('Test Error')
a = clock;
figname = ['netflix_100M-testErrs-mu'  num2str(mu*100) '-' date '-' num2str(a(4)) '-'  num2str(a(5))];
savefig(fig2, figname, 'Figures');
close all;


end

